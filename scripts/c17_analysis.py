"""Addendum 5, Sections 8-9: the ONE pre-specified analysis of the Camelyon17-WILDS replication.

Reads saved results only (no training). Prints and saves:
  1. integrity checks (all runs present, frozen model unchanged, metrics recomputed from logits)
  2. per cell: drift-attributable dBrier, dMCB, DSC lost under FT (seed table with test patients),
     R1, R2 (Fieller), R3, cell status; overall verdict from the two primary cells
  3. descriptive: rho bootstrap (share of non-positive draws), dMCB upper bound vs SESOI,
     per-test-patient dBrier, target hospital (G) FT vs replay, source-model AUROC
  4. exploratory frozen-BatchNorm control with absolute end-of-stream Brier and twin change
  5. optional, only if declared in Addendum 5 Section 12 before running: --insample_sensitivity

Usage (Colab, CPU is enough):
  python scripts/c17_analysis.py --data /content/c17data --models $D2/models --out $D2/stream_runs \
      --splits $D2/splits
Writes <out>/analysis_c17/ (report.txt and CSV tables).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from calcl import c17_stats as St  # noqa: E402
from calcl.corp import corp_crossfit, corp_insample  # noqa: E402

CELLS = ["c17_1to4", "c17_0to3", "c17_4to1", "c17_3to0"]
PRIMARY = ["c17_1to4", "c17_0to3"]
SEEDS = [1, 2, 3, 4, 5]
RUNS = ["frozen_gradual", "ft_gradual", "ft_nodrift", "replay_gradual", "replay_nodrift",
        "ft_gradual_bnfrozen", "ft_nodrift_bnfrozen"]


class Report:
    def __init__(self):
        self.lines = []

    def __call__(self, s=""):
        print(s, flush=True)
        self.lines.append(s)


def rdir(a, cell, seed):
    return os.path.join(a.out, f"{cell}_resnet18_seed{seed}")


def metrics(a, cell, seed, run):
    m = pd.read_csv(os.path.join(rdir(a, cell, seed), f"stream_{run}", "metrics.csv"))
    return m


def s_end_start(m, setname="S", cols=("brier", "mcb", "dsc")):
    s = m[m.set == setname].set_index("checkpoint").sort_index()
    return {k: (float(s[k].iloc[0]), float(s[k].iloc[-1])) for k in cols}


def drift(a, cell, seed, strat, tag=""):
    g = s_end_start(metrics(a, cell, seed, f"{strat}_gradual{tag}"))
    n = s_end_start(metrics(a, cell, seed, f"{strat}_nodrift{tag}"))
    ch = lambda d, k: d[k][1] - d[k][0]  # noqa: E731
    return {"d_brier": ch(g, "brier") - ch(n, "brier"), "d_mcb": ch(g, "mcb") - ch(n, "mcb"),
            "dsc_lost": -(ch(g, "dsc") - ch(n, "dsc")),
            "end_brier_gradual": g["brier"][1], "end_brier_twin": n["brier"][1],
            "start_brier": g["brier"][0], "twin_change": ch(n, "brier")}


def check_inventory(a, lab):
    """Every expected source model and stream run, complete and as specified. Returns a list of problems."""
    problems = []
    for c in CELLS:
        for s in SEEDS:
            src = os.path.join(a.models, f"{c}_resnet18_seed{s}")
            for f in ("summary.json", "source.pt", "eval_idx.npz"):
                if not os.path.exists(os.path.join(src, f)):
                    problems.append(f"source model {c} seed {s}: missing {f}")
            for r in RUNS:
                d = os.path.join(rdir(a, c, s), f"stream_{r}")
                need = [f for f in ("metrics.csv", "logits.npz", "meta.json") if not os.path.exists(os.path.join(d, f))]
                if need:
                    problems.append(f"{c} seed {s} {r}: missing {', '.join(need)}")
                    continue
                m = pd.read_csv(os.path.join(d, "metrics.csv"))
                sm = m[m.set == "S"]
                n_ck = sm.checkpoint.nunique()
                if n_ck != a.checkpoints:
                    problems.append(f"{c} seed {s} {r}: {n_ck} checkpoints, expected {a.checkpoints}")
                if not np.isfinite(sm[["brier", "mcb", "dsc"]].to_numpy(float)).all():
                    problems.append(f"{c} seed {s} {r}: non-finite S metrics")
                meta = json.load(open(os.path.join(d, "meta.json")))
                if bool(meta.get("freeze_bn", False)) != r.endswith("_bnfrozen"):
                    problems.append(f"{c} seed {s} {r}: freeze_bn flag is {meta.get('freeze_bn')}")
                if not r.startswith("frozen") and abs(float(meta.get("lr", -1)) - 0.001) > 1e-12:
                    problems.append(f"{c} seed {s} {r}: lr {meta.get('lr')}, expected 0.001")
                lg = np.load(os.path.join(d, "logits.npz"))
                if "idx_B" in lg.files:
                    problems.append(f"{c} seed {s} {r}: unexpected buffer set B")
                if sum(k.startswith("S_c") for k in lg.files) != a.checkpoints:
                    problems.append(f"{c} seed {s} {r}: logits saved for the wrong number of checkpoints")
    return problems


def probs(z, T0):
    return softmax(np.asarray(z, np.float64) / T0, axis=1)[:, 1]


def fmt(t, d=4):
    return f"{t[0]:+.{d}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


def main(a):
    global CELLS, PRIMARY, SEEDS
    CELLS, PRIMARY = a.cells.split(","), a.primary.split(",")
    SEEDS = [int(x) for x in a.seeds.split(",")]
    R = Report()
    man = pd.read_csv(os.path.join(a.data, "c17_compact_manifest.csv"))
    lab, pat, wsi = man.label.to_numpy(), man.patient.to_numpy(), man.wsi.to_numpy()
    od = os.path.join(a.out, "analysis_c17")
    if os.path.exists(os.path.join(od, "report.txt")) and not a.allow_rerun:
        sys.exit(f"{od}/report.txt already exists: the pre-specified analysis has been run once already. "
                 "Not running again (Addendum 5). Use --allow_rerun only to reproduce it, and say so in the paper.")
    os.makedirs(od, exist_ok=True)

    # ---------------- 1. inventory and integrity ----------------
    R("=== 1. Inventory and integrity ===")
    R("Primary decomposition: cross-fitted CORP over 2 slide folds (saved metrics.csv). All criteria use it.")
    problems = check_inventory(a, lab)
    n_src, n_std, n_bn = len(CELLS) * len(SEEDS), len(CELLS) * len(SEEDS) * 5, len(CELLS) * len(SEEDS) * 2
    if problems:
        R(f"INVENTORY FAILED ({len(problems)} problems). No result is computed:")
        for p_ in problems:
            R(f"  - {p_}")
        open(os.path.join(od, "inventory_failed.txt"), "w").write("\n".join(R.lines))
        sys.exit(1)
    R(f"expected and found: {n_src} source models, {n_std} standard stream runs, {n_bn} frozen-BatchNorm runs "
      f"({n_std + n_bn} stream runs in total); every run complete (11 checkpoints, finite S metrics and logits, "
      "correct freeze_bn flag and learning rate)")
    worst_recompute, worst_frozen = 0.0, 0.0
    T0 = {}
    for c in CELLS:
        for s in SEEDS:
            T0[(c, s)] = json.load(open(os.path.join(a.models, f"{c}_resnet18_seed{s}", "summary.json")))["T0"]
            fz = s_end_start(metrics(a, c, s, "frozen_gradual"))
            worst_frozen = max(worst_frozen, abs(fz["brier"][1] - fz["brier"][0]))
            for r in ("ft_gradual", "ft_nodrift"):
                lg = np.load(os.path.join(rdir(a, c, s), f"stream_{r}", "logits.npz"))
                idx = lg["idx_S"]
                m = metrics(a, c, s, r)
                for ck in (0, int(m.checkpoint.max())):
                    re_ = corp_crossfit(probs(lg[f"S_c{ck}"], T0[(c, s)]), lab[idx], wsi[idx])["brier"]
                    saved = float(m[(m.set == "S") & (m.checkpoint == ck)].brier.iloc[0])
                    worst_recompute = max(worst_recompute, abs(re_ - saved))
    R(f"frozen model: largest change in S Brier over the stream = {worst_frozen:.2e} (should be ~0)")
    R(f"S Brier recomputed from saved logits vs saved metrics: largest difference {worst_recompute:.2e} "
      "(float16 logits; should be < 1e-3)")
    assert worst_frozen < 1e-4 and worst_recompute < 1e-3

    # ---------------- 2. criteria ----------------
    R("\n=== 2. Replication criteria (Addendum 5, Section 9) ===")
    seed_rows, cell_rows, status = [], [], {}
    for c in CELLS:
        ft = [drift(a, c, s, "ft") for s in SEEDS]
        rp = [drift(a, c, s, "replay") for s in SEEDS]
        arr = {k: np.array([d[k] for d in ft]) for k in ("d_brier", "d_mcb", "dsc_lost")}
        res = St.cell_criteria(arr, replay_dsc_lost=np.array([d["dsc_lost"] for d in rp]))
        status[c] = res["status"]
        lo, hi, kind = res["fieller"]
        blo, bhi, share, flag = res["bootstrap"]
        R(f"\n--- {c} ({'PRIMARY' if c in PRIMARY else 'secondary'}) ---")
        R(f"{'seed':>4} {'dBrier':>8} {'dMCB':>8} {'DSClost':>8} {'replay DSClost':>14}  source test patients | target test patients")
        for s, f_, r_ in zip(SEEDS, ft, rp):
            info = json.load(open(os.path.join(a.splits, f"{c}_seed{s}.json")))
            st_ = [p for p, x in info["src_roles"].items() if x == "test"]
            tt_ = [p for p, x in info["tgt_roles"].items() if x == "test"]
            R(f"{s:>4} {f_['d_brier']:+8.4f} {f_['d_mcb']:+8.4f} {f_['dsc_lost']:+8.4f} {r_['dsc_lost']:+14.4f}  "
              f"{' '.join(st_)} | {' '.join(tt_)}")
            seed_rows.append({"cell": c, "seed": s, **{f"ft_{k}": v for k, v in f_.items()},
                              **{f"replay_{k}": v for k, v in r_.items()},
                              "src_test_patients": " ".join(st_), "tgt_test_patients": " ".join(tt_)})
        R(f"FT dBrier   {fmt(res['dBrier'])}")
        R(f"FT dMCB     {fmt(res['dMCB'])}   (upper bound vs SESOI 0.005: {'below' if res['dMCB'][2] < St.SESOI else 'not below'})")
        R(f"FT DSC lost {fmt(res['DSC_lost'])}")
        R(f"rho = {res['rho']:+.3f}; Fieller 95% {'[' + f'{lo:+.3f}, {hi:+.3f}' + ']' if kind == 'bounded' else 'unbounded (not estimable)'}; "
          f"contrast dMCB - 0.25 dBrier {fmt(res['L025'])}")
        R(f"  descriptive bootstrap: [{blo:+.3f}, {bhi:+.3f}], non-positive denominators {share:.1%} ({flag})")
        R(f"R1 {'holds' if res['R1'] else 'fails'} | R2 {'holds' if res['R2'] else 'fails'} | "
          f"R3 (directional): replay below FT in {res['R3_seeds']}/{len(SEEDS)} seeds, paired FT - replay DSC lost "
          f"{fmt(res['R3_paired_FT_minus_replay'])} | STATUS: {res['status']}")
        cell_rows.append({"cell": c, "primary": c in PRIMARY, "status": res["status"], "R1": res["R1"], "R2": res["R2"],
                          "R3_seeds": res["R3_seeds"], "dBrier": res["dBrier"], "dMCB": res["dMCB"],
                          "DSC_lost": res["DSC_lost"], "rho": res["rho"], "fieller_lo": lo, "fieller_hi": hi,
                          "fieller_kind": kind, "boot_lo": blo, "boot_hi": bhi, "boot_nonpos_share": share})
    verdict = St.overall_verdict([status[c] for c in PRIMARY])
    R(f"\nOVERALL VERDICT (primary cells {', '.join(PRIMARY)}; cross-fitted, primary analysis): {verdict.upper()}")
    pd.DataFrame(seed_rows).to_csv(os.path.join(od, "seeds.csv"), index=False)
    pd.DataFrame(cell_rows).to_csv(os.path.join(od, "cells.csv"), index=False)

    # ---------------- 3. descriptive ----------------
    R("\n=== 3. Descriptive (uncorrected) ===")
    R("Per-test-patient drift-attributable dBrier on S (FT), mean over the seeds in which the patient was tested:")
    pp = []
    for c in CELLS:
        for s in SEEDS:
            L = {r: np.load(os.path.join(rdir(a, c, s), f"stream_ft_{r}", "logits.npz")) for r in ("gradual", "nodrift")}
            idx = L["gradual"]["idx_S"]
            y = lab[idx]
            last = max(int(k.split("_c")[1]) for k in L["gradual"].files if k.startswith("S_c"))
            for p in np.unique(pat[idx]):
                k = pat[idx] == p
                b = {r: [np.mean((probs(L[r][f"S_c{ck}"], T0[(c, s)])[k] - y[k]) ** 2) for ck in (0, last)] for r in L}
                pp.append({"cell": c, "seed": s, "patient": int(p), "n": int(k.sum()), "tumour": int(y[k].sum()),
                           "d_brier": (b["gradual"][1] - b["gradual"][0]) - (b["nodrift"][1] - b["nodrift"][0])})
    pp = pd.DataFrame(pp)
    pp.to_csv(os.path.join(od, "per_patient.csv"), index=False)
    R(pp.groupby(["cell", "patient"]).agg(seeds=("seed", "nunique"), tumour=("tumour", "max"),
                                          d_brier=("d_brier", "mean")).round(4).to_string())
    R("\nTarget hospital (G) at end of stream, FT minus replay, paired over seeds:")
    for c in CELLS:
        diffs = {"auroc": [], "brier": []}
        for s in SEEDS:
            e = {r: metrics(a, c, s, f"{r}_gradual") for r in ("ft", "replay")}
            for k in diffs:
                g = {r: float(e[r][(e[r].set == "G") & (e[r].checkpoint == e[r].checkpoint.max())][k].iloc[0]) for r in e}
                diffs[k].append(g["ft"] - g["replay"])
        R(f"  {c}: AUROC {fmt(St.tci(diffs['auroc']))} | Brier {fmt(St.tci(diffs['brier']))}")
    R(f"\nSource-model AUROC on S before the stream (all {len(CELLS) * len(SEEDS)} models):")
    src = pd.DataFrame([{"cell": c, "seed": s, "S_auroc": json.load(open(os.path.join(
        a.models, f"{c}_resnet18_seed{s}", "summary.json")))["S_source_test"]["auroc"]} for c in CELLS for s in SEEDS])
    src.to_csv(os.path.join(od, "source_auroc.csv"), index=False)
    R(src.pivot(index="cell", columns="seed", values="S_auroc").round(3).to_string())
    if src.S_auroc.min() < 0.75:
        R("  note: at least one source model has S AUROC below 0.75 (reported, not excluded; Addendum 5, Section 10)")

    # ---------------- 4. frozen-BN control ----------------
    R("\n=== 4. Exploratory: frozen-BatchNorm control (Addendum 5, Section 7) ===")
    bn_rows = []
    for c in CELLS:
        ft = [drift(a, c, s, "ft") for s in SEEDS]
        bn = [drift(a, c, s, "ft", "_bnfrozen") for s in SEEDS]
        g = lambda L, k: np.array([d[k] for d in L])  # noqa: E731
        R(f"--- {c} ---")
        R(f"  frozen-BN drift-attributable: dBrier {fmt(St.tci(g(bn, 'd_brier')))} | dMCB {fmt(St.tci(g(bn, 'd_mcb')))} | "
          f"DSC lost {fmt(St.tci(g(bn, 'dsc_lost')))}")
        R(f"  paired FT - frozen-BN DSC lost {fmt(St.tci(g(ft, 'dsc_lost') - g(bn, 'dsc_lost')))}")
        R(f"  absolute end Brier on S: FT gradual {g(ft, 'end_brier_gradual').mean():.4f}, FT twin "
          f"{g(ft, 'end_brier_twin').mean():.4f}, BN gradual {g(bn, 'end_brier_gradual').mean():.4f}, BN twin "
          f"{g(bn, 'end_brier_twin').mean():.4f}; start {g(ft, 'start_brier').mean():.4f}")
        R(f"  twin's own change (end - start): FT {fmt(St.tci(g(ft, 'twin_change')))} | BN {fmt(St.tci(g(bn, 'twin_change')))}")
        fix = St.tci(g(bn, "end_brier_gradual") - g(ft, "end_brier_gradual"))
        R(f"  BN gradual - FT gradual end Brier {fmt(fix)} -> "
          f"{'may be called a fix (CI below 0)' if fix[2] < 0 else 'NOT a fix by the pre-specified rule'}")
        for s, f_, b_ in zip(SEEDS, ft, bn):
            bn_rows.append({"cell": c, "seed": s, **{f"ft_{k}": v for k, v in f_.items()}, **{f"bn_{k}": v for k, v in b_.items()}})
    pd.DataFrame(bn_rows).to_csv(os.path.join(od, "bn_control.csv"), index=False)

    # ---------------- 5. optional declared sensitivity ----------------
    if a.insample_sensitivity:
        R("\n=== 5. Secondary, exploratory sensitivity: in-sample CORP on S (Addendum 5, Section 12, entry 1) ===")
        R("Fitted and evaluated on the same patches, so it may be optimistic. It does not change any criterion, "
          "status or the verdict above.")
        ins_seed, ins_cell = [], []
        prim = {r_["cell"]: r_ for r_ in cell_rows}
        for c in CELLS:
            rows = []
            for s in SEEDS:
                d = {}
                for r in ("gradual", "nodrift"):
                    lg = np.load(os.path.join(rdir(a, c, s), f"stream_ft_{r}", "logits.npz"))
                    y = lab[lg["idx_S"]]
                    last = max(int(k.split("_c")[1]) for k in lg.files if k.startswith("S_c"))
                    d[r] = [corp_insample(probs(lg[f"S_c{ck}"], T0[(c, s)]), y) for ck in (0, last)]
                ch = lambda r, k: d[r][1][k] - d[r][0][k]  # noqa: E731
                rows.append({"d_brier": ch("gradual", "brier") - ch("nodrift", "brier"),
                             "d_mcb": ch("gradual", "mcb") - ch("nodrift", "mcb"),
                             "dsc_lost": -(ch("gradual", "dsc") - ch("nodrift", "dsc"))})
                ins_seed.append({"cell": c, "seed": s, **rows[-1]})
            arr = {k: np.array([x[k] for x in rows]) for k in rows[0]}
            res = St.cell_criteria(arr)
            lo, hi, kind = res["fieller"]
            pr = prim[c]
            side = lambda r_: ("n/a" if not np.isfinite(r_) else ">= 0.25" if r_ >= 0.25 else "< 0.25")  # noqa: E731
            agree = (res["status"] == pr["status"]) and (side(res["rho"]) == side(pr["rho"]))
            R(f"  {c}: dMCB {fmt(res['dMCB'])}, DSC lost {fmt(res['DSC_lost'])}, rho {res['rho']:+.3f}, "
              f"Fieller {'[' + f'{lo:+.3f}, {hi:+.3f}' + ']' if kind == 'bounded' else 'unbounded'} -> status {res['status']}")
            R(f"     primary (cross-fitted): rho {pr['rho']:+.3f}, status {pr['status']} -> "
              + ("AGREE" if agree else f"DISAGREE (status {pr['status']} vs {res['status']}; rho {side(pr['rho'])} vs {side(res['rho'])})"))
            ins_cell.append({"cell": c, "status_insample": res["status"], "status_primary": pr["status"],
                             "rho_insample": res["rho"], "rho_primary": pr["rho"], "fieller_lo": lo, "fieller_hi": hi,
                             "dMCB": res["dMCB"], "DSC_lost": res["DSC_lost"], "agree": agree})
        pd.DataFrame(ins_seed).to_csv(os.path.join(od, "insample_seeds.csv"), index=False)
        pd.DataFrame(ins_cell).to_csv(os.path.join(od, "insample_cells.csv"), index=False)
        dis = [x["cell"] for x in ins_cell if not x["agree"]]
        R(f"  disagreement between in-sample and primary: {', '.join(dis) if dis else 'none'} "
          "(the primary result stands in every case)")

    open(os.path.join(od, "report.txt"), "w").write("\n".join(R.lines))
    R(f"\nSaved in {od}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--splits", required=True)
    p.add_argument("--cells", default=",".join(CELLS), help="leave at the default (Addendum 5); for tests only")
    p.add_argument("--primary", default=",".join(PRIMARY), help="leave at the default (Addendum 5); for tests only")
    p.add_argument("--seeds", default=",".join(map(str, SEEDS)), help="leave at the default (Addendum 5); for tests only")
    p.add_argument("--checkpoints", type=int, default=11, help="checkpoints per run (10 windows + start)")
    p.add_argument("--allow_rerun", action="store_true", help="reproduce an analysis already run once")
    p.add_argument("--insample_sensitivity", action="store_true",
                   help="only if declared in Addendum 5, Section 12, before this script is run")
    main(p.parse_args())
