import numpy as np
import pandas as pd
import pytest

from calcl import c17_split as C


def synth_meta(seed=0, step=32):
    """WILDS-like metadata: patches on a 32-unit grid; per slide a tissue rectangle with a tumour blob."""
    rng = np.random.default_rng(seed)
    rows, slide = [], 0
    spec = {0: [4, 9, 10, 12, 15], 1: [20, 21, 22, 34, 36], 3: [60, 61, 62, 73, 75], 4: [80, 81, 92, 96, 99]}
    for c, pats in spec.items():
        for i, p in enumerate(pats):
            rich = i >= 2
            for node in range(2 if p in (15, 20) else 1):
                w, h = 140, 140  # grid positions
                xs, ys = np.meshgrid(np.arange(w), np.arange(h))
                xs, ys = xs.ravel(), ys.ravel()
                keep = rng.random(xs.size) < 0.6
                xs, ys = xs[keep], ys[keep]
                r = 45 if rich else 6
                tum = (xs - 70) ** 2 + (ys - 70) ** 2 < r ** 2
                rows.append(pd.DataFrame({"patient": p, "node": node, "x_coord": xs * step, "y_coord": ys * step,
                                          "tumor": tum.astype(int), "slide": slide, "center": c}))
                slide += 1
    return pd.concat(rows, ignore_index=True)


def test_measure_units_recovers_scale():
    rng = np.random.default_rng(0)
    big = (rng.random((2000, 2000, 3)) * 255).astype(np.uint8)
    for upx in (1, 2):  # coordinate units per pixel
        meta = pd.DataFrame({"slide": 0, "x_coord": np.arange(0, 1500, 32) * upx, "y_coord": 64 * upx})
        pairs = C.find_neighbour_pairs(meta, 10)
        load = lambda s, x, y: big[y // upx:y // upx + 96, x // upx:x // upx + 96]  # noqa: E731
        u = C.measure_units(pairs, load)
        assert u["px_per_unit"] == pytest.approx(1 / upx)
        assert u["P_units"] == pytest.approx(96 * upx)
        assert u["step_px"] == 32


def test_measure_units_rejects_unrelated_images():
    rng = np.random.default_rng(1)
    meta = pd.DataFrame({"slide": 0, "x_coord": np.arange(0, 1500, 32), "y_coord": 0})
    pairs = C.find_neighbour_pairs(meta, 10)
    with pytest.raises(AssertionError):
        C.measure_units(pairs, lambda s, x, y: (rng.random((96, 96, 3)) * 255).astype(np.uint8))


def test_overlap_pairs_definition():
    a = pd.DataFrame({"slide": [0], "x_coord": [0], "y_coord": [0]})
    for dx, dy, expect in [(95, 0, 1), (96, 0, 0), (0, 95, 1), (95, 95, 1), (96, 95, 0), (-95, 0, 1)]:
        b = pd.DataFrame({"slide": [0], "x_coord": [dx], "y_coord": [dy]})
        assert C.overlap_pairs(a, b, 96) == expect
    assert C.overlap_pairs(a, pd.DataFrame({"slide": [1], "x_coord": [0], "y_coord": [0]}), 96) == 0


def test_cap_is_deterministic_and_capped():
    m = synth_meta()
    c1, c2 = C.cap_patches(m, cap=500), C.cap_patches(m, cap=500)
    assert c1.index.equals(c2.index)
    assert c1.groupby(["slide", "tumor"]).size().max() <= 500


def test_spatial_split_has_no_cross_subset_overlap():
    m = C.cap_patches(synth_meta(), cap=3000)
    tr = m[m.patient.isin([12, 15])]
    lab, rep = C.spatial_split(tr, 96, seed=3)
    subs = {s: tr[lab == s] for s in ("train", "val_es", "val_temp")}
    assert all(len(v) for v in subs.values())
    for a, b in (("train", "val_es"), ("train", "val_temp"), ("val_es", "val_temp")):
        assert C.overlap_pairs(subs[a], subs[b], 96) == 0
    # removal is exactly the patches that touched another subset; report adds up
    assigned = sum(rep[f"{s}_assigned_t"] + rep[f"{s}_assigned_n"] for s in subs)
    assert assigned == len(tr)
    assert sum(rep[f"{s}_removed"] for s in subs) == int((lab == "removed").sum())
    lab2, _ = C.spatial_split(tr, 96, seed=3)
    assert lab.equals(lab2)


def test_leakage_assertion_catches_injected_overlap():
    m = C.cap_patches(synth_meta(), cap=3000)
    tr = m[m.patient.isin([12, 15])]
    lab, _ = C.spatial_split(tr, 96, seed=3)
    src = {12: "train", 15: "train", 10: "stream", 4: "test", 9: "test"}
    tgt = {60: "stream", 62: "test"}
    C.assert_no_leakage(m, src, tgt, lab, 96)  # clean split passes
    # move one early-stopping patch into the temperature subset: it overlaps its val_es neighbours
    bad = lab.copy()
    bad[bad.index[bad == "val_es"][len(bad[bad == "val_es"]) // 2]] = "val_temp"
    with pytest.raises(AssertionError):
        C.assert_no_leakage(m, src, tgt, bad, 96)
    # a test patient used for validation
    with pytest.raises(AssertionError):
        C.assert_no_leakage(m, {**src, 12: "test"}, tgt, lab, 96)
    # a patient in both hospitals
    with pytest.raises(AssertionError):
        C.assert_no_leakage(m, src, {**tgt, 12: "stream"}, lab, 96)


def test_plan_cell_is_deterministic_and_feasible_on_synthetic_data():
    m = C.cap_patches(synth_meta(), cap=3000)
    a, b = C.plan_cell(m, "c17_1to4", 2, 96), C.plan_cell(m, "c17_1to4", 2, 96)
    assert a["feasible"] and a["src_roles"] == b["src_roles"] and a["tgt_roles"] == b["tgt_roles"]
    for s, req in C.FINAL.items():
        assert a["val"][f"{s}_t"] >= req and a["val"][f"{s}_n"] >= req
    roles = set(a["src_roles"].values())
    assert roles == {"train", "stream", "test"}


def test_feasibility_script_end_to_end(tmp_path):
    import importlib.util, os
    spec = importlib.util.spec_from_file_location("feas", os.path.join(os.path.dirname(__file__), "..", "scripts", "c17_feasibility.py"))
    F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
    rng = np.random.default_rng(0)
    big = (rng.random((5000, 5000, 3)) * 255).astype(np.uint8)
    meta = synth_meta()
    t = F.run(meta, lambda s, x, y: big[y:y + 96, x:x + 96], str(tmp_path), seeds=[1, 2])
    assert len(t) == 8 and (tmp_path / "feasibility_report.txt").exists()


def test_arrangements_are_distinct_role_sets():
    for rich in ([1, 2, 3], [1, 2, 3, 4]):
        for side in ("src", "tgt"):
            arrs = C.arrangements(rich, side, 2026)
            assert len({tuple(sorted(a.items())) for a in arrs}) == len(arrs)
    assert len(C.arrangements([1, 2, 3, 4], "tgt", 2027)) == 6   # {2 stream, 2 test} partitions
    assert len(C.arrangements([1, 2, 3, 4], "src", 2026)) == 12


def test_seeds_rotate_through_distinct_arrangements():
    m = C.cap_patches(synth_meta(), cap=3000)
    rs = C.plan_cell_seeds(m, "c17_1to4", [1, 2, 3], 96)
    assert all(r["feasible"] for r in rs)
    assert len({r["src_info"]["arrangement"] for r in rs}) == 3
