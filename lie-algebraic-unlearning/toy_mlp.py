"""Reproduce the toy-MLP geometry + retain-constraint experiment (paper Section 6.9, Figure 2)
and SAVE Exp2_1.png (geometry preservation) and Exp2_2.png (forget-retain tradeoff).

Adapted from Cell 3 of Lie_Group_Orthogonal_Pipeline.ipynb: logic unchanged, `display` -> print,
`plt.show()` -> savefig into the manuscript directory. MNIST downloads to ./data on first run.

Run: python toy_mlp.py   (CPU, ~3-5 min incl. MNIST download)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device, dtype = "cpu", torch.float32
EPS = 1e-8

BATCH_SIZE, BASE_EPOCHS, LR_BASE, LR_ADAPT = 128, 5, 1e-3, 5e-3
HIDDEN, RANK_LORA = 256, 8
FORGET_CLASS, SCRAMBLED_TARGET = 7, 1
MAX_PER_CLASS_BASE, MAX_FORGET, MAX_RETAIN = 1200, 1000, 2000
FORGET_TARGET_LOSS, MAX_ADAPT_STEPS = 0.05, 300
GEOM_NORMS, GEOM_SAMPLES = [0.01, 0.05, 0.1, 0.2, 0.4], 20


def make_skew(M): return 0.5 * (M - M.T)
def normalize_fro(A, t): return A * (t / (torch.norm(A, p="fro") + EPS))
def random_skew(n, t): return normalize_fro(make_skew(torch.randn(n, n, device=device, dtype=dtype)), t)
def gram_delta(Wb, Wa):
    G0, G1 = Wb.T @ Wb, Wa.T @ Wa
    return (torch.norm(G0 - G1, p="fro") / (torch.norm(G0, p="fro") + EPS)).item()


def filter_by_classes(ds, classes, mpc):
    counts = {c: 0 for c in classes}; idxs = []
    for i, (_, y) in enumerate(ds):
        if y in classes and counts[y] < mpc:
            idxs.append(i); counts[y] += 1
    return idxs


def filter_indices(ds, fn, mx):
    idxs = []
    for i, (_, y) in enumerate(ds):
        if fn(y):
            idxs.append(i)
            if len(idxs) >= mx: break
    return idxs


tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tfm)

base_idx = filter_by_classes(train_ds, list(range(10)), MAX_PER_CLASS_BASE)
forget_idx = filter_indices(train_ds, lambda y: y == FORGET_CLASS, MAX_FORGET)
retain_idx = filter_indices(train_ds, lambda y: y != FORGET_CLASS, MAX_RETAIN)
forget_test_idx = filter_indices(test_ds, lambda y: y == FORGET_CLASS, 500)
retain_test_idx = filter_indices(test_ds, lambda y: y != FORGET_CLASS, 2000)

base_loader = DataLoader(Subset(train_ds, base_idx), batch_size=BATCH_SIZE, shuffle=True)
forget_loader = DataLoader(Subset(train_ds, forget_idx), batch_size=BATCH_SIZE, shuffle=True)
retain_loader = DataLoader(Subset(train_ds, retain_idx), batch_size=BATCH_SIZE, shuffle=True)
forget_test_loader = DataLoader(Subset(test_ds, forget_test_idx), batch_size=BATCH_SIZE)
retain_test_loader = DataLoader(Subset(test_ds, retain_test_idx), batch_size=BATCH_SIZE)


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, HIDDEN, bias=False)
        self.fc2 = nn.Linear(HIDDEN, 10)
    def forward_with_weight(self, x, W):
        return self.fc2(F.relu(F.linear(x.view(x.size(0), -1), W)))
    def forward(self, x):
        return self.forward_with_weight(x, self.fc1.weight)


base = ToyMLP().to(device)
opt = optim.Adam(base.parameters(), lr=LR_BASE)
print("training base MLP...")
for ep in range(BASE_EPOCHS):
    for x, y in base_loader:
        opt.zero_grad(); loss = F.cross_entropy(base(x), y); loss.backward(); opt.step()
    print(f"  epoch {ep+1} loss {loss.item():.4f}")
for p in base.parameters(): p.requires_grad = False
W0 = base.fc1.weight.detach().clone(); n, d = W0.shape

# ---- Part A: geometry ----
print("geometry experiment...")
geom = {m: [] for m in ("additive_low_rank", "first_order_skew", "exact_exp_skew")}
for t in GEOM_NORMS:
    accA = {m: [] for m in geom}
    for _ in range(GEOM_SAMPLES):
        A = random_skew(n, t)
        W_exp = torch.matrix_exp(A) @ W0
        W_first = (torch.eye(n, device=device, dtype=dtype) + A) @ W0
        B = torch.randn(n, RANK_LORA, device=device, dtype=dtype)
        C = torch.randn(RANK_LORA, d, device=device, dtype=dtype)
        Delta = B @ C
        Delta = Delta * (torch.norm(W_exp - W0, p="fro") / (torch.norm(Delta, p="fro") + EPS))
        accA["additive_low_rank"].append(gram_delta(W0, W0 + Delta))
        accA["first_order_skew"].append(gram_delta(W0, W_first))
        accA["exact_exp_skew"].append(gram_delta(W0, W_exp))
    for m in geom: geom[m].append(np.mean(accA[m]))

plt.figure(figsize=(8, 5))
for m in geom: plt.plot(GEOM_NORMS, geom[m], "o-", label=m)
plt.xlabel("Generator Frobenius norm"); plt.ylabel("Mean normalized Gram deviation")
plt.title("Toy Network: Geometry Preservation"); plt.yscale("log")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Exp2_1.png"), dpi=150); plt.close()
print("  exact_exp_skew gram dev:", [f"{v:.1e}" for v in geom["exact_exp_skew"]])
print("  additive gram dev:", [f"{v:.1e}" for v in geom["additive_low_rank"]])


@torch.no_grad()
def eval_w(W, loader, mode="normal"):
    tot_l = tot_c = tot = 0
    for x, y in loader:
        y_ev = torch.full_like(y, SCRAMBLED_TARGET) if mode == "forget" else y
        logits = base.forward_with_weight(x, W)
        tot_l += F.cross_entropy(logits, y_ev, reduction="sum").item()
        tot_c += (logits.argmax(-1) == y_ev).sum().item(); tot += x.size(0)
    return {"loss": tot_l / tot, "acc": tot_c / tot}


def retain_basis(Wb, nb=20):
    S = []; it = iter(retain_loader)
    for _ in range(nb):
        try: x, y = next(it)
        except StopIteration: it = iter(retain_loader); x, y = next(it)
        W = Wb.detach().clone().requires_grad_(True)
        loss = F.cross_entropy(base.forward_with_weight(x, W), y)
        Gr = torch.autograd.grad(loss, W)[0]
        Sr = make_skew((0.5 * (Gr @ Wb.T - Wb @ Gr.T)).detach())
        if torch.norm(Sr, p="fro") > EPS: S.append(Sr.reshape(-1))
    U, s, _ = torch.linalg.svd(torch.stack(S, 1), full_matrices=False)
    energy = torch.cumsum(s ** 2, 0) / (torch.sum(s ** 2) + EPS)
    k = int((energy < 0.95).sum().item()) + 1
    print(f"  retain basis rank k={k}, energy={energy[k-1]:.3f}")
    return U[:, :k].detach()


def proj_svd(G, Uk):
    g = G.reshape(-1); return (g - Uk @ (Uk.T @ g)).reshape_as(G)
def align(A, Uk):
    a = A.reshape(-1); return (torch.norm(Uk @ (Uk.T @ a)) / (torch.norm(a) + EPS)).item()


class Additive(nn.Module):
    def __init__(s, W0, r=8):
        super().__init__(); s.W0 = W0.detach().clone()
        s.B = nn.Parameter(0.01 * torch.randn(n, r, device=device))
        s.C = nn.Parameter(0.01 * torch.randn(r, d, device=device))
    def W_eff(s): return s.W0 + s.B @ s.C

class FirstOrder(nn.Module):
    def __init__(s, W0):
        super().__init__(); s.W0 = W0.detach().clone()
        s.M = nn.Parameter(0.001 * torch.randn(n, n, device=device))
    def A(s): return make_skew(s.M)
    def W_eff(s): return (torch.eye(n, device=device) + s.A()) @ s.W0

class ExpSkew(nn.Module):
    def __init__(s, W0):
        super().__init__(); s.W0 = W0.detach().clone()
        s.M = nn.Parameter(0.001 * torch.randn(n, n, device=device))
    def A(s): return make_skew(s.M)
    def W_eff(s): return torch.matrix_exp(s.A()) @ s.W0


def train_forget(adapter, name, Uk=None, projected=False):
    adapter = adapter.to(device); opt = optim.Adam(adapter.parameters(), lr=LR_ADAPT)
    it = iter(forget_loader)
    for step in range(MAX_ADAPT_STEPS):
        try: x, y = next(it)
        except StopIteration: it = iter(forget_loader); x, y = next(it)
        yt = torch.full_like(y, SCRAMBLED_TARGET)
        opt.zero_grad()
        loss = F.cross_entropy(base.forward_with_weight(x, adapter.W_eff()), yt)
        loss.backward()
        if projected and hasattr(adapter, "M") and Uk is not None:
            with torch.no_grad():
                adapter.M.grad.copy_(proj_svd(make_skew(adapter.M.grad), Uk))
        opt.step()
        if loss.item() <= FORGET_TARGET_LOSS: break
    We = adapter.W_eff().detach()
    fa = eval_w(We, forget_test_loader, "forget"); rb = eval_w(W0, retain_test_loader)
    ra = eval_w(We, retain_test_loader)
    al = align(adapter.A().detach(), Uk) if (hasattr(adapter, "A") and Uk is not None) else np.nan
    return {"method": name, "forget_loss_after": fa["loss"],
            "retain_delta_loss": ra["loss"] - rb["loss"], "gram_delta": gram_delta(W0, We),
            "alignment": al}


print("retain-constraint experiment...")
Uk = retain_basis(W0, 20)
res = [
    train_forget(Additive(W0, RANK_LORA), "lora_additive"),
    train_forget(FirstOrder(W0), "first_order_skew", Uk, False),
    train_forget(ExpSkew(W0), "exact_exp_skew_unprojected", Uk, False),
    train_forget(ExpSkew(W0), "exact_exp_skew_projected", Uk, True),
]
for r in res:
    print(f"  {r['method']:28s} forget={r['forget_loss_after']:.3f} "
          f"retain_delta={r['retain_delta_loss']:.3f} gram={r['gram_delta']:.2e} "
          f"align={r['alignment']:.3f}")

plt.figure(figsize=(7, 5))
plt.scatter([r["forget_loss_after"] for r in res], [r["retain_delta_loss"] for r in res])
for r in res:
    plt.annotate(r["method"], (r["forget_loss_after"], r["retain_delta_loss"]), fontsize=8)
plt.xlabel("Final forget loss"); plt.ylabel("Retain loss degradation")
plt.title("Toy Network: Forget-Retain Tradeoff"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Exp2_2.png"), dpi=150); plt.close()

print("\nsaved:", [os.path.join(FIG_DIR, f) for f in ("Exp2_1.png", "Exp2_2.png")])
