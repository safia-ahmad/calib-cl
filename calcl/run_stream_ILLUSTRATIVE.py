"""ILLUSTRATIVE, UNTESTED skeleton of the continual-learning loop (PyTorch).

It was written without a GPU or PyTorch available, so treat it as a blueprint:
read it, then implement and unit-test each piece yourself. metrics.py,
stream.py and alarms.py ARE tested (see tests/).

Protocol per window t (test-then-train, boundary-free):
  1. evaluate theta_{t-1} on E_t (fresh current-distribution images), S (fixed
     source test), G (fixed target test)  -> one row per (t, eval set)
  2. update on window W_t with the strategy; the strategy never sees t, alpha_t,
     or domain ids; its hyper-parameters are fixed before the stream starts
"""
import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from calcl.metrics import all_metrics


# ------------------------------------------------------------------ strategies
class Strategy:
    def __init__(self, model, lr=0.01, wd=5e-4):
        self.model = model
        self.opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                                   weight_decay=wd)

    def loss(self, x, y):
        return F.cross_entropy(self.model(x), y)

    def observe(self, x, y):              # one mini-batch from the stream
        self.model.train()
        self.opt.zero_grad()
        self.loss(x, y).backward()
        self.opt.step()
        self.after_step(x, y)

    def after_step(self, x, y):
        pass


class SequentialFineTune(Strategy):
    pass


class ExperienceReplay(Strategy):
    """Reservoir buffer; each step mixes current and replayed samples 1:1."""

    def __init__(self, model, buffer_size=1000, **kw):
        super().__init__(model, **kw)
        self.size, self.seen, self.bx, self.by = buffer_size, 0, [], []

    def loss(self, x, y):
        if self.bx:
            k = min(len(self.bx), len(x))
            j = random.sample(range(len(self.bx)), k)
            x = torch.cat([x, torch.stack([self.bx[i] for i in j]).to(x.device)])
            y = torch.cat([y, torch.stack([self.by[i] for i in j]).to(y.device)])
        return F.cross_entropy(self.model(x), y)

    def after_step(self, x, y):
        for xi, yi in zip(x.detach().cpu(), y.detach().cpu()):
            self.seen += 1
            if len(self.bx) < self.size:
                self.bx.append(xi); self.by.append(yi)
            else:
                r = random.randrange(self.seen)
                if r < self.size:
                    self.bx[r], self.by[r] = xi, yi


class OnlineEWC(Strategy):
    """Online EWC with consolidation every `every` steps (a schedule fixed in
    advance, NOT at domain boundaries, so it stays boundary-free)."""

    def __init__(self, model, lam=100.0, gamma=0.9, every=200, **kw):
        super().__init__(model, **kw)
        self.lam, self.gamma, self.every, self.steps = lam, gamma, every, 0
        self.fisher = None
        self.anchor = None
        self.recent = []                  # last mini-batches for Fisher estimate

    def loss(self, x, y):
        base = F.cross_entropy(self.model(x), y)
        if self.fisher is None:
            return base
        pen = sum((f * (p - a) ** 2).sum() for f, p, a in
                  zip(self.fisher, self.model.parameters(), self.anchor))
        return base + 0.5 * self.lam * pen

    def after_step(self, x, y):
        self.steps += 1
        self.recent = (self.recent + [(x.detach(), y.detach())])[-10:]
        if self.steps % self.every == 0:
            new = [torch.zeros_like(p) for p in self.model.parameters()]
            self.model.eval()
            for xb, yb in self.recent:    # empirical Fisher on recent data
                self.model.zero_grad()
                F.cross_entropy(self.model(xb), yb).backward()
                for n_, p in zip(new, self.model.parameters()):
                    if p.grad is not None:
                        n_ += p.grad.detach() ** 2 / len(self.recent)
            self.fisher = new if self.fisher is None else [
                self.gamma * f + n_ for f, n_ in zip(self.fisher, new)]
            self.anchor = [p.detach().clone() for p in self.model.parameters()]


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def logits_on(model, loader, device):
    model.eval()
    zs, ys = [], []
    for x, y in loader:
        zs.append(model(x.to(device)).float().cpu()); ys.append(y)
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def run_stream(model, strategy, stream_windows, make_loader, eval_sets, T0,
               device, batch_size=64, passes=1, seed=0, log=print):
    """stream_windows: list of index arrays (from calcl.stream.build_stream)
    make_loader(idx, shuffle): DataLoader over those dataset indices
    eval_sets: dict name -> callable(t) returning a DataLoader (E_t depends on t;
               S and G ignore t)
    T0: temperature fitted once on source validation and then kept fixed."""
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rows = []
    for t, w in enumerate(stream_windows, start=1):
        for name, get in eval_sets.items():           # 1) test ...
            z, y = logits_on(model, get(t), device)
            rows.append({"t": t, "eval": name, **all_metrics(z, y, deployed_t=T0)})
        for _ in range(passes):                        # 2) ... then train
            for x, y in make_loader(w, shuffle=True):
                strategy.observe(x.to(device), y.to(device))
        torch.save(copy.deepcopy(model.state_dict()), f"ckpt_t{t:03d}.pt")
        log(f"window {t} done")
    for name, get in eval_sets.items():                # final checkpoint
        z, y = logits_on(model, get(len(stream_windows) + 1), device)
        rows.append({"t": len(stream_windows) + 1, "eval": name,
                     **all_metrics(z, y, deployed_t=T0)})
    return rows
