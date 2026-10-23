"""Day 8-10: train one source model, fit its temperature, and score it on S and G.

Usage (from the repo root):
  python scripts/train_source.py --setting pcam_fwd --seed 0 --model resnet18 \
      --data /content/data --out /content/drive/MyDrive/calib-cl-data/runs

Writes to <out>/<setting>_<model>_seed<seed>/:
  split.csv        slide -> role for this seed
  eval_idx.npz     fixed evaluation patch indices (S, G)
  source.pt        best source weights
  summary.json     T0, training history, metrics on S and G, and gate G4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calcl.corp import corp_crossfit  # noqa: E402
from calcl.data import indices_for, load_compact, slide_table  # noqa: E402
from calcl.metrics import all_metrics, fit_temperature  # noqa: E402
from calcl.models import make_model  # noqa: E402
from calcl.splits import eval_sample, make_split, summarise  # noqa: E402
from calcl.train import make_loader, predict, seed_everything, train_source  # noqa: E402

SESOI = 0.005


def score(z, y, groups, T0):
    p1 = softmax(z / T0, axis=1)[:, 1]
    out = {k: float(v) for k, v in all_metrics(z, y, deployed_t=T0).items()}
    out.pop("brier", None)  # multi-class form; the binary Brier from CORP is used instead
    out.update({k: float(v) for k, v in corp_crossfit(p1, y, groups).items()})
    return out


def main(a):
    t_start = time.time()
    seed_everything(a.seed)
    run = os.path.join(a.out, f"{a.setting}_{a.model}_seed{a.seed}")
    os.makedirs(run, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| run folder:", run)

    X, man = load_compact(os.path.join(a.data, "pcam_compact.h5"),
                          os.path.join(a.data, "pcam_compact_manifest.csv"))
    split = make_split(slide_table(man), a.setting, a.seed)
    split.to_csv(os.path.join(run, "split.csv"), index=False)
    print(summarise(split))

    def slides(domain, role):
        return split[(split.domain == domain) & (split.role == role)].wsi

    tr = indices_for(man, slides("source", "train"))
    es = indices_for(man, slides("source", "val_es"))
    vt = indices_for(man, slides("source", "val_temp"))
    S = eval_sample(man, slides("source", "test"), n=a.n_eval, seed=a.seed)
    G = eval_sample(man, slides("target", "test"), n=a.n_eval, seed=a.seed)
    np.savez(os.path.join(run, "eval_idx.npz"), S=S, G=G)
    if a.max_train:
        tr = np.random.default_rng(a.seed).choice(tr, min(a.max_train, len(tr)), replace=False)
    print(f"patches: train {len(tr)}, val_es {len(es)}, val_temp {len(vt)}, S {len(S)}, G {len(G)}")

    y = man.label.to_numpy()
    model = make_model(a.model, pretrained=not a.no_pretrained)
    res = train_source(model,
                       make_loader(X, y, tr, train=True, batch=a.batch, workers=a.workers, seed=a.seed),
                       make_loader(X, y, es, train=False, batch=256, workers=a.workers),
                       dev, lr=a.lr, max_epochs=a.epochs, patience=a.patience)
    torch.save(model.state_dict(), os.path.join(run, "source.pt"))

    z_vt, y_vt = predict(model, make_loader(X, y, vt, train=False, batch=256, workers=a.workers), dev)
    T0 = fit_temperature(z_vt, y_vt)
    print(f"temperature T0 = {T0:.3f}")

    out = {"setting": a.setting, "seed": a.seed, "model": a.model, "T0": T0, **res}
    for name, idx in [("S_source_test", S), ("G_target_test", G)]:
        z, yy = predict(model, make_loader(X, y, idx, train=False, batch=256, workers=a.workers), dev)
        out[name] = score(z, yy, man.wsi.to_numpy()[idx], T0)
        np.savez(os.path.join(run, f"logits_{name}.npz"), z=z, y=yy, idx=idx)
    gap = out["G_target_test"]["brier"] - out["S_source_test"]["brier"]
    out["G4_brier_gap"] = gap
    out["G4_pass"] = bool(gap >= SESOI)
    out["minutes"] = (time.time() - t_start) / 60
    with open(os.path.join(run, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== results (temperature-scaled) ===")
    for name in ["S_source_test", "G_target_test"]:
        r = out[name]
        print(f"{name:14s} acc {r['acc']:.3f}  AUROC {r['auroc']:.3f}  Brier {r['brier']:.4f}  "
              f"MCB {r['mcb']:.4f}  DSC {r['dsc']:.4f}  UNC {r['unc']:.4f}")
    print(f"G4: Brier(G) - Brier(S) = {gap:.4f} -> {'PASS' if out['G4_pass'] else 'FAIL'} (needs >= {SESOI})")
    print(f"total time {out['minutes']:.1f} min")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--setting", default="pcam_fwd", choices=["pcam_fwd", "pcam_rev"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "vit_s16"])
    p.add_argument("--data", default="/content/data")
    p.add_argument("--out", default="/content/drive/MyDrive/calib-cl-data/runs")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--n_eval", type=int, default=2000)
    p.add_argument("--max_train", type=int, default=0, help="cap on training patches (0 = all); for quick tests")
    p.add_argument("--no_pretrained", action="store_true", help="for offline tests only")
    main(p.parse_args())
