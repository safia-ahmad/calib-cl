"""Day 16-17: run the stream experiments for one setting, model and seed.

Usage (from the repo root), e.g. on Kaggle:
  python scripts/run_stream.py --setting pcam_fwd --seed 0 --model resnet18 \
      --data $DATA --runs /kaggle/working/runs --streams /kaggle/working/streams \
      --strategies frozen,ft,replay --kinds gradual,nodrift

Needs, from earlier steps:
  <runs>/<setting>_<model>_seed<seed>/  source.pt, summary.json (T0), eval_idx.npz (S, G), split.csv
  <streams>/<setting>_seed<seed>.npz     gradual / nodrift windows
Writes into the run folder, or into <out>/<run folder name>/ when --out is given:
  stream_<strategy>_<kind>/metrics.csv      one row per checkpoint x evaluation set
  stream_<strategy>_<kind>/logits.npz       logits on S, G, B at every checkpoint (for the fixes)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl.data import load_compact  # noqa: E402
from calcl.models import make_model  # noqa: E402
from calcl.splits import eval_sample  # noqa: E402
from calcl.stream_train import Reservoir, score_set, train_window  # noqa: E402
from calcl.train import make_loader, predict, seed_everything  # noqa: E402


def run_one(a, X, man, run_dir, strategy, kind, dev):
    s = json.load(open(os.path.join(run_dir, "summary.json")))
    T0 = s["T0"]
    ev = np.load(os.path.join(run_dir, "eval_idx.npz"))
    split = pd.read_csv(os.path.join(run_dir, "split.csv"))
    B = eval_sample(man, split[(split.domain == "source") & (split.role == "val_buf")].wsi,
                    n=a.n_buffer, seed=a.seed)
    windows = np.load(os.path.join(a.streams, f"{a.setting}_seed{a.seed}.npz"))[kind]
    sets = {"S": ev["S"], "G": ev["G"], "B": B}
    y_all, wsi = man.label.to_numpy(), man.wsi.to_numpy()

    seed_everything(1000 + a.seed)
    rng = np.random.default_rng(1000 + a.seed)
    model = make_model(a.model, pretrained=False)
    model.load_state_dict(torch.load(os.path.join(run_dir, "source.pt"), map_location="cpu"))
    model.to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, weight_decay=5e-4)
    buffer = Reservoir(a.buffer, rng) if strategy == "replay" else None

    out_dir = os.path.join(a.out or run_dir, os.path.basename(run_dir), f"stream_{strategy}_{kind}") \
        if a.out else os.path.join(run_dir, f"stream_{strategy}_{kind}")
    os.makedirs(out_dir, exist_ok=True)
    rows, logits = [], {}
    T = len(windows)
    t_start = time.time()
    for c in range(T + 1):  # checkpoint c = windows trained so far
        evals = dict(sets)
        if c < T:
            evals["E"] = windows[c]
        for name, idx in evals.items():
            z, y = predict(model, make_loader(X, y_all, idx, train=False, batch=256, workers=a.workers), dev)
            r = score_set(z, y, wsi[idx], T0)
            rows.append({"setting": a.setting, "seed": a.seed, "model": a.model, "strategy": strategy,
                         "stream": kind, "checkpoint": c, "set": name, "n": len(idx), **r})
            if name != "E":
                logits[f"{name}_c{c}"] = z.astype(np.float16)
        if c < T and strategy != "frozen":
            loss = train_window(model, opt, X, y_all, windows[c], dev, rng, batch=a.batch,
                                buffer=buffer, colour=a.colour)
            rows[-1]["train_loss_next"] = loss
        last = {r_["set"]: r_ for r_ in rows if r_["checkpoint"] == c}
        print(f"  c={c:2d} | S brier {last['S']['brier']:.4f} mcb {last['S']['mcb']:.4f} "
              f"dsc {last['S']['dsc']:.4f} | G brier {last['G']['brier']:.4f} auroc {last['G']['auroc']:.3f}",
              flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "logits.npz"),
                        **{f"idx_{k}": v for k, v in sets.items()}, **logits)
    meta = {"T0": T0, "lr": a.lr, "batch": a.batch, "buffer": a.buffer if buffer else 0,
            "colour": a.colour, "minutes": (time.time() - t_start) / 60,
            "buffer_items": len(buffer.items) if buffer else 0}
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=2)
    return meta


def main(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, man = load_compact(os.path.join(a.data, "pcam_compact.h5"),
                          os.path.join(a.data, "pcam_compact_manifest.csv"))
    run_dir = os.path.join(a.runs, f"{a.setting}_{a.model}_seed{a.seed}")
    for strategy in a.strategies.split(","):
        for kind in a.kinds.split(","):
            if strategy == "frozen" and kind == "nodrift":
                continue  # frozen never trains, so the twin would be identical
            print(f"=== {a.setting} {a.model} seed {a.seed} | {strategy} on {kind} stream ===", flush=True)
            meta = run_one(a, X, man, run_dir, strategy, kind, dev)
            print(f"  done in {meta['minutes']:.1f} min" +
                  (f", buffer holds {meta['buffer_items']}" if meta["buffer_items"] else ""))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="pcam_fwd", choices=["pcam_fwd", "pcam_rev"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "vit_s16"])
    p.add_argument("--data", default="/content/data")
    p.add_argument("--runs", default="/kaggle/working/runs")
    p.add_argument("--out", default="", help="where to write results (default: inside the run folder; "
                   "needed when the runs folder is read-only, e.g. a Kaggle input)")
    p.add_argument("--streams", default="/kaggle/working/streams")
    p.add_argument("--strategies", default="frozen,ft,replay")
    p.add_argument("--kinds", default="gradual,nodrift")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--buffer", type=int, default=1000)
    p.add_argument("--n_buffer", type=int, default=1000, help="size of the labelled source set B")
    p.add_argument("--colour", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=2)
    main(p.parse_args())
