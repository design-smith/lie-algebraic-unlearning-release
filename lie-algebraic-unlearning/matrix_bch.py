"""Reproduce the matrix-level BCH validation (paper Section 6.8, Table 1, Figure 1) and SAVE
the three figures the manuscript references: Exp1_1.png, Exp1_2.png, Exp1_3.png.

Adapted from Cell 2 of Lie_Group_Orthogonal_Pipeline.ipynb: the logic is unchanged, `display`
is replaced by print, `plt.show()` by `savefig`, and figures are written to the manuscript
directory so the `![](Exp1_1.png)` references resolve.

Run: python matrix_bch.py   (CPU, ~1-2 min)
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# figures/ subdirectory of the repo root
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cpu"
dtype = torch.float64
EPS = 1e-12

N = 128
N_SAMPLES = 200
NORM_LEVELS = [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]


def make_skew(M):
    return 0.5 * (M - M.T)


def normalize_fro(A, t):
    return A * (t / (torch.norm(A, p="fro") + EPS))


def random_skew(n, t):
    return normalize_fro(make_skew(torch.randn(n, n, device=device, dtype=dtype)), t)


def commutator(A, B):
    return A @ B - B @ A


def fro(A):
    return torch.norm(A, p="fro").item()


rows = []
for target_norm in NORM_LEVELS:
    print(f"norm level {target_norm}")
    for _ in range(N_SAMPLES):
        A1, A2 = random_skew(N, target_norm), random_skew(N, target_norm)
        R_exact = torch.matrix_exp(A2) @ torch.matrix_exp(A1)
        R_naive = torch.matrix_exp(A1 + A2)
        C = commutator(A2, A1)
        R_bch = torch.matrix_exp(make_skew(A1 + A2 + 0.5 * C))
        dn, db = fro(R_exact - R_naive), fro(R_exact - R_bch)
        denom = fro(R_exact) + EPS
        rows.append((target_norm, dn, db, dn / denom, db / denom, dn / (db + EPS), fro(C)))

cols = ["target_norm", "naive", "bch", "naive_n", "bch_n", "improve", "comm"]
arr = np.array(rows)
summary = {}
for i, t in enumerate(NORM_LEVELS):
    m = arr[:, 0] == t
    summary[t] = {c: arr[m, j].mean() for j, c in enumerate(cols)}

print("\nTable 1 (matrix-level BCH composition accuracy):")
print(f"{'norm':>6} {'naive_mean':>12} {'bch_mean':>12} {'improvement':>12}")
xs = NORM_LEVELS
naive_m = [summary[t]["naive"] for t in xs]
bch_m = [summary[t]["bch"] for t in xs]
naive_n = [summary[t]["naive_n"] for t in xs]
bch_n = [summary[t]["bch_n"] for t in xs]
improve = [summary[t]["improve"] for t in xs]
for t in xs:
    s = summary[t]
    print(f"{t:6.2f} {s['naive']:12.3e} {s['bch']:12.3e} {s['improve']:11.1f}x")

# Figure 1(a): absolute composition error
plt.figure(figsize=(8, 5))
plt.plot(xs, naive_m, "o-", label="Naive")
plt.plot(xs, bch_m, "o-", label="Second-order BCH")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Generator Frobenius norm"); plt.ylabel("Mean composition error")
plt.title("BCH vs Naive Composition Error"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Exp1_1.png"), dpi=150); plt.close()

# Figure 1(b): normalized composition error
plt.figure(figsize=(8, 5))
plt.plot(xs, naive_n, "o-", label="Naive normalized")
plt.plot(xs, bch_n, "o-", label="BCH normalized")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Generator Frobenius norm"); plt.ylabel("Normalized mean composition error")
plt.title("Normalized BCH vs Naive Composition Error"); plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Exp1_2.png"), dpi=150); plt.close()

# Figure 1(c): improvement factor
plt.figure(figsize=(8, 5))
plt.plot(xs, improve, "o-")
plt.xscale("log"); plt.yscale("log")
plt.xlabel("Generator Frobenius norm"); plt.ylabel("Mean improvement factor (naive / BCH)")
plt.title("BCH Improvement Factor Across Norm Regimes"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Exp1_3.png"), dpi=150); plt.close()

print("\nsaved:", [os.path.join(FIG_DIR, f) for f in ("Exp1_1.png", "Exp1_2.png", "Exp1_3.png")])
