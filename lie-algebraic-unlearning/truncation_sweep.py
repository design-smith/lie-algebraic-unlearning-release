"""Addendum M: matrix-level rank-truncation sweep for Theorem 1 (paper Section 6.3, Figure 1(d)).

Sweeps the truncation rank R at fixed generator norm in so(128) and measures the composition
error of rank-truncated second-order BCH accumulation against the two-term Theorem 1 bound. The
bound is evaluated in its small-angle (L = 1) form: the running-rho BCH remainder sum_t beta_t
plus the measured Eckart-Young tails sum_t tau_t. The script also records
max_t(eps + ||H_t||) -- the quantity that must stay below ln 2 for Theorem 1's hypothesis to
hold -- and locates where it crosses ln 2, so the figure can mark the theorem's regime of
validity. It emits Exp1_4.png in the manuscript directory.

This is the generating code for Figure 1(d) (Reproducibility, Protocol Rule 1). Reimplemented
from the paper's description; numbers match the independent check to three figures.

Run: python truncation_sweep.py    (CPU, ~2 min; requires numpy, scipy, matplotlib)
"""

import os
import numpy as np
from scipy.linalg import expm, logm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
n = 128
r_task = 4                       # per-task A = U V^T - V U^T, U,V in R^{n x 4}, rank <= 8
LN2 = np.log(2.0)


def rand_gen(eps, rng):
    U = rng.standard_normal((n, r_task))
    V = rng.standard_normal((n, r_task))
    A = U @ V.T - V @ U.T
    return A * (eps / np.linalg.norm(A))


def skew_trunc(B, R):
    """Best rank-2R Frobenius approximation, re-projected to skew (matches the real-Schur
    truncation to Eckart-Young precision). Returns (B_trunc, tau)."""
    if R is None:
        return 0.5 * (B - B.T), 0.0
    U, s, Vt = np.linalg.svd(B)
    k = 2 * R
    Bt = (U[:, :k] * s[:k]) @ Vt[:k, :]
    Bt = 0.5 * (Bt - Bt.T)
    return Bt, float(np.linalg.norm(B - Bt))


def run(eps, T, R, seed):
    rng = np.random.default_rng(seed)
    A_list = [rand_gen(eps, rng) for _ in range(T)]
    H = np.zeros((n, n))
    P = np.eye(n)
    sum_tau = sum_beta = max_rho = 0.0
    for A in A_list:
        rho_t = np.linalg.norm(H)
        sum_beta += (1.0 / 3.0) * eps * rho_t * (eps + rho_t)   # running-rho 3rd-order BCH term
        B = H + A + 0.5 * (A @ H - H @ A)
        H, tau = skew_trunc(B, R)
        sum_tau += tau
        P = expm(A) @ P
        max_rho = max(max_rho, np.linalg.norm(H))
    exact_rho = np.linalg.norm(logm(P).real)                    # exact accumulated generator
    max_eh = eps + max(max_rho, exact_rho)
    realized = float(np.linalg.norm(P - expm(H)))
    return realized, sum_tau, sum_beta, sum_beta + sum_tau, max_eh


def crossing_eps(T, eps_grid):
    """eps where max_t(eps + ||H_t||) crosses ln 2 (using the untruncated history, the largest)."""
    eh = [np.mean([run(e, T, None, s)[4] for s in (1, 2, 3)]) for e in eps_grid]
    for i in range(len(eps_grid) - 1):
        if eh[i] < LN2 <= eh[i + 1]:
            return float(np.interp(LN2, [eh[i], eh[i + 1]], [eps_grid[i], eps_grid[i + 1]]))
    return None


# ---------------- table ----------------
print(f"{'eps':>5} {'T':>3} {'R':>5} | {'realized':>9} {'sum_tau':>8} {'bound(L=1)':>10} "
      f"{'e+rho':>6} {'regime':>7}")
any_violation = False
for T in (8, 16):
    for eps in (0.01, 0.10, 0.40):
        for R in (4, 8, 16, None):
            res = np.mean([run(eps, T, R, s) for s in (1, 2, 3)], axis=0)
            reg = "in" if res[4] < LN2 else "OUT"
            any_violation |= res[0] > res[3]
            print(f"{eps:>5} {T:>3} {str(R):>5} | {res[0]:9.3g} {res[1]:8.3g} {res[3]:10.3g} "
                  f"{res[4]:6.3f} {reg:>7}")
    print()
print("bound (L=1 form) dominated realized error on all printed cells."
      if not any_violation else "WARNING: a cell exceeded the small-angle bound.")

# ---------------- Figure 1(d): T=16, realized vs small-angle bound, per R, with ln2 marker ----
T_fig = 16
eps_grid = np.array([0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
colors = {4: "C0", 8: "C1", 16: "C2", None: "C3"}
fig, ax = plt.subplots(figsize=(7, 5))
for R in (4, 8, 16, None):
    realized, bound = [], []
    for e in eps_grid:
        res = np.mean([run(e, T_fig, R, s) for s in (1, 2, 3)], axis=0)
        realized.append(res[0]); bound.append(res[3])
    ax.loglog(eps_grid, realized, "o-", color=colors[R], label=f"realized R={R}")
    ax.loglog(eps_grid, bound, "--", color=colors[R], alpha=0.55, label=f"bound R={R}")

xc8 = crossing_eps(8, eps_grid)
xc16 = crossing_eps(16, eps_grid)
if xc16:
    ax.axvline(xc16, color="k", ls=":", lw=1.2)
    ax.text(xc16 * 1.03, ax.get_ylim()[0] * 3, r"$\varepsilon+\rho=\ln 2$ (T=16)",
            rotation=90, va="bottom", fontsize=7)
ax.set_xlabel(r"generator Frobenius norm $\varepsilon$")
ax.set_ylabel("Frobenius composition error")
ax.set_title(f"Rank-truncated composition error vs small-angle Theorem 1 bound (T={T_fig})")
ax.legend(fontsize=6, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "Exp1_4.png"), dpi=150)
print(f"\nln2-crossing eps: T=8 ~ {xc8:.3f}, T=16 ~ {xc16:.3f}")
print("saved", os.path.join(FIG_DIR, "Exp1_4.png"))
