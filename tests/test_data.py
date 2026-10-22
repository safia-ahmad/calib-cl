import h5py
import numpy as np
import pandas as pd
import pytest

from calcl.data import (PatchDataset, indices_for, load_compact, mean_colour,
                        normalise, slide_table)


@pytest.fixture
def tiny(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    X = rng.integers(0, 256, size=(n, 96, 96, 3), dtype=np.uint8)
    man = pd.DataFrame({
        "wsi": [f"camelyon16_train_{'tumor' if i % 4 == 0 else 'normal'}_{i // 4:03d}" for i in range(n)],
        "image": [f"{'tumor' if i % 4 == 0 else 'normal'}_{i // 4:03d}.tif" for i in range(n)],
        "hospital": ["RUMC" if i < 20 else "UMCU" for i in range(n)],
        "label": [int(i % 4 == 0 and i % 8 == 0) for i in range(n)],
    })
    h5 = tmp_path / "c.h5"
    with h5py.File(h5, "w") as f:
        f["x"] = X
        f["y"] = man["label"].to_numpy().astype(np.uint8)
    csv = tmp_path / "m.csv"
    man.to_csv(csv, index=False)
    return X, man, str(h5), str(csv)


def test_load_and_checks(tiny):
    X, man, h5, csv = tiny
    X2, man2 = load_compact(h5, csv)
    assert X2.shape == X.shape and (X2 == X).all()
    assert set(man2.slide_type) == {"tumour", "normal"}


def test_label_mismatch_is_caught(tiny, tmp_path):
    X, man, h5, csv = tiny
    bad = man.copy()
    bad.loc[0, "label"] = 1 - bad.loc[0, "label"]
    p = tmp_path / "bad.csv"
    bad.to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_compact(h5, str(p))


def test_dataset_item_and_augmentation(tiny):
    X, man, h5, csv = tiny
    X2, man2 = load_compact(h5, csv)
    ds = PatchDataset(X2, man2.label.to_numpy(), np.arange(10), train=True)
    img, lab = ds[3]
    assert img.shape == (3, 96, 96) and img.dtype == np.float32
    assert lab == man2.label[3]
    plain = PatchDataset(X2, man2.label.to_numpy(), np.arange(10), train=False)
    np.testing.assert_allclose(plain[3][0], normalise(X2[3]))


def test_slide_helpers(tiny):
    X, man, h5, csv = tiny
    X2, man2 = load_compact(h5, csv)
    st = slide_table(man2)
    assert len(st) == man2.wsi.nunique()
    one = st.wsi.iloc[0]
    idx = indices_for(man2, [one])
    assert set(man2.wsi.iloc[idx]) == {one}
    assert (man2.label.iloc[indices_for(man2, st.wsi, label=1)] == 1).all()
    assert mean_colour(X2, np.arange(10)).shape == (3,)
