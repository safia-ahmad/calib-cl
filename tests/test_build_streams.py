import importlib.util
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("bs", os.path.join(ROOT, "scripts", "build_streams.py"))
bs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bs)


def _man():
    rows = []
    for h, nn, nt in [("RUMC", 100, 70), ("UMCU", 59, 40)]:
        for t, n in [("normal", nn), ("tumor", nt)]:
            for i in range(n):
                for k in range(120):
                    rows.append({"wsi": f"{t}_{h}{i}", "image": f"{t}_{h}{i}.tif", "hospital": h,
                                 "label": int(t == "tumor" and k % 2 == 0)})
    m = pd.DataFrame(rows)
    m["slide_type"] = np.where(m.image.str.startswith("tumor"), "tumour", "normal")
    return m


def test_streams_follow_ramp_and_never_reuse():
    man = _man()
    st = bs.make_streams(man, "pcam_fwd", seed=1, T=10, n=100, tumour_share=0.5)
    for key in ["gradual", "nodrift"]:
        flat = st[key].ravel()
        assert len(flat) == len(set(flat)) == 1000
    assert st["target_share"][0] == 0 and st["target_share"][-1] == 1
    assert np.all(np.diff(st["target_share"][1:-1]) > -0.25)  # rises, up to sampling noise
    nod_hosp = man.hospital.to_numpy()[st["nodrift"].ravel()]
    assert (nod_hosp == "RUMC").all()
