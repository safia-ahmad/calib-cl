"""Data access for the compact PCam file built on Day 4.

The whole image array (about 3.6 GB as uint8) is loaded into RAM once; this is
much faster than reading gzip-compressed HDF5 chunks at random during training.
No PyTorch import is needed here: PatchDataset works with torch.utils.data.DataLoader
because it has __len__ and __getitem__, and DataLoader turns the numpy arrays
it returns into tensors. Covered by tests/test_data.py.
"""
from __future__ import annotations

import random

import h5py
import numpy as np
import pandas as pd

# ImageNet statistics, because the models start from ImageNet weights.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_compact(h5_path: str, manifest_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load all images into RAM and check they line up with the manifest."""
    man = pd.read_csv(manifest_path)
    with h5py.File(h5_path, "r") as f:
        X = f["x"][:]
        Y = f["y"][:]
    if len(X) != len(man):
        raise ValueError(f"{len(X)} images but {len(man)} manifest rows")
    if not (Y == man["label"].to_numpy()).all():
        raise ValueError("labels in the .h5 file and the manifest disagree")
    man = man.copy()
    man["slide_type"] = np.where(man["image"].str.startswith("tumor"), "tumour", "normal")
    return X, man


def slide_table(man: pd.DataFrame) -> pd.DataFrame:
    """One row per slide: hospital, slide type, number of patches, tumour patches."""
    return (man.groupby("wsi")
               .agg(hospital=("hospital", "first"), slide_type=("slide_type", "first"),
                    n_patches=("label", "size"), n_tumour=("label", "sum"))
               .reset_index())


def indices_for(man: pd.DataFrame, slides, label: int | None = None) -> np.ndarray:
    """Row indices of all patches from the given slides (optionally one class)."""
    mask = man["wsi"].isin(set(slides))
    if label is not None:
        mask &= man["label"] == label
    return np.flatnonzero(mask.to_numpy())


def normalise(img_uint8: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 3xHxW float32, ImageNet-normalised."""
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def augment(img: np.ndarray) -> np.ndarray:
    """Random 90-degree rotation and flips. Safe for histology (no preferred
    orientation). Uses the `random` module, which PyTorch reseeds in each
    DataLoader worker, so workers do not repeat the same augmentations."""
    k = random.randint(0, 3)
    img = np.rot90(img, k, axes=(0, 1))
    if random.random() < 0.5:
        img = img[:, ::-1]
    return img


def colour_jitter(img: np.ndarray, s: float) -> np.ndarray:
    """Random stain-like colour change of strength s (0 = none, 1 = strong):
    per-channel gain (+-15% * s), brightness (+-10% * s), contrast and
    saturation (+-20% * s). uint8 in, uint8 out. Uses `random`, like augment."""
    if s <= 0:
        return img
    x = img.astype(np.float32)
    gain = np.array([1 + random.uniform(-0.15, 0.15) * s for _ in range(3)], dtype=np.float32)
    x = x * gain
    x = x + random.uniform(-0.10, 0.10) * s * 255
    m = x.mean()
    x = m + (x - m) * (1 + random.uniform(-0.2, 0.2) * s)
    grey = x.mean(axis=2, keepdims=True)
    x = grey + (x - grey) * (1 + random.uniform(-0.2, 0.2) * s)
    return np.clip(x, 0, 255).astype(np.uint8)


class PatchDataset:
    """Patches selected by row index. Returns (3x96x96 float32, int label).
    colour > 0 adds colour_jitter to training patches (off by default)."""

    def __init__(self, X: np.ndarray, labels: np.ndarray, indices, train: bool = False,
                 colour: float = 0.0):
        self.X = X
        self.labels = np.asarray(labels)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.train = train
        self.colour = colour

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        img = self.X[j]
        if self.train:
            img = colour_jitter(augment(img), self.colour)
        return normalise(img), int(self.labels[j])


def mean_colour(X: np.ndarray, idx: np.ndarray, max_n: int = 5000, seed: int = 0) -> np.ndarray:
    """Average R, G, B (0-255) over up to max_n patches: a simple number for the colour shift."""
    rng = np.random.default_rng(seed)
    pick = idx if len(idx) <= max_n else rng.choice(idx, max_n, replace=False)
    return X[np.sort(pick)].reshape(-1, 3).mean(0)
