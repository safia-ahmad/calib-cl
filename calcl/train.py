"""Source-model training and prediction (PyTorch). Covered by tests/test_train.py,
which runs a tiny model on random data on CPU."""
from __future__ import annotations

import copy
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from calcl.data import PatchDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X, labels, idx, train: bool, batch: int = 128, workers: int = 2, seed: int = 0,
                colour: float = 0.0):
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(PatchDataset(X, labels, idx, train=train, colour=colour), batch_size=batch,
                      shuffle=train, num_workers=workers, generator=g,
                      pin_memory=torch.cuda.is_available(), drop_last=False)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Logits (N, 2) and labels (N,), in loader order."""
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            z = model(x.to(device, non_blocking=True))
        zs.append(z.float().cpu())
        ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def train_source(model, train_loader, es_loader, device, lr: float = 1e-4, wd: float = 1e-4,
                 max_epochs: int = 10, patience: int = 3, log=print) -> dict:
    """AdamW training with early stopping on validation NLL. Returns the best
    weights (already loaded into `model`) and the per-epoch history."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * len(y)
            n += len(y)
        z, y_es = predict(model, es_loader, device)
        val_nll = float(F.cross_entropy(torch.from_numpy(z), torch.from_numpy(y_es)).item())
        val_acc = float((z.argmax(1) == y_es).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_nll": val_nll,
                        "val_acc": val_acc, "seconds": time.time() - t0})
        log(f"epoch {epoch}: train loss {tot / n:.4f} | val NLL {val_nll:.4f} | val acc {val_acc:.3f} "
            f"| {time.time() - t0:.0f}s")
        if val_nll < best - 1e-4:
            best, bad = val_nll, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            bad += 1
            if bad >= patience:
                log(f"early stop at epoch {epoch}; best val NLL {best:.4f}")
                break
    model.load_state_dict(best_state)
    best_epoch = min(history, key=lambda h: h["val_nll"])["epoch"]
    return {"best_epoch": best_epoch, "best_val_nll": best, "history": history}
