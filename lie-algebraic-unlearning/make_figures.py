"""Regenerate every figure in the paper with one command.

    python make_figures.py

Figures 1(a-c), 1(d), and 2(a-b) are recomputed from scratch on CPU. Figure 3 (Exp_H1.png) is
rebuilt from results/experiment1_results.csv, the committed output of the H1 experiment
(h1_llama.py, run on a GPU); its Spearman rho is recomputed from the same rows so the figure
cannot drift from the data. All seven PNGs land in figures/ under their exact paper filenames.
Exits non-zero if any output is missing or stale.
"""
import csv
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")

JOBS = [
    ("matrix_bch.py", ["Exp1_1.png", "Exp1_2.png", "Exp1_3.png"], "Figure 1(a-c) matrix BCH"),
    ("truncation_sweep.py", ["Exp1_4.png"], "Figure 1(d) truncation sweep"),
    ("toy_mlp.py", ["Exp2_1.png", "Exp2_2.png"], "Figure 2(a-b) toy MLP (downloads MNIST)"),
]


def make_fig3():
    """Figure 3: commutator vs interference, rebuilt from the committed H1 results CSV."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "experiment1_results.csv"),
                                    encoding="utf-8")))
    k = np.array([float(r["kappa"]) for r in rows])
    intf = np.array([float(r["interference"]) for r in rows])
    rho, p = spearmanr(k, intf)
    plt.figure(figsize=(6, 5))
    for cond, col in [("entity", "C0"), ("attribute", "C3")]:
        rs = [r for r in rows if r["condition"] == cond]
        plt.scatter([float(r["kappa"]) for r in rs], [float(r["interference"]) for r in rs],
                    c=col, label=cond, alpha=0.7)
    plt.xlabel(r"normalized commutator $\hat\kappa_{ij}$")
    plt.ylabel("interference(i, j): resurfaced answer probability")
    plt.title(f"H1: Spearman rho = {rho:.2f} (p={p:.1e})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "Exp_H1.png"), dpi=150)
    plt.close()
    print(f"  wrote Exp_H1.png  (rho={rho:.3f}, p={p:.2e}, n={len(rows)})")


def run():
    os.makedirs(FIG, exist_ok=True)
    ok = True
    for script, outputs, label in JOBS:
        print(f"\n=== {label}: {script} ===", flush=True)
        start = time.time()
        before = {o: os.path.getmtime(os.path.join(FIG, o)) if os.path.exists(os.path.join(FIG, o))
                  else 0 for o in outputs}
        proc = subprocess.run([sys.executable, os.path.join(ROOT, script)], cwd=ROOT)
        if proc.returncode != 0:
            print(f"  FAILED: {script} exited {proc.returncode}")
            ok = False
            continue
        for o in outputs:
            path = os.path.join(FIG, o)
            if not os.path.exists(path) or os.path.getmtime(path) <= before[o]:
                print(f"  MISSING/STALE: {o}")
                ok = False
            else:
                print(f"  wrote {o}")
        print(f"  ({time.time() - start:.0f}s)")
    print("\n=== Figure 3 H1 (from results CSV) ===")
    try:
        make_fig3()
    except Exception as e:
        print("  FAILED:", e)
        ok = False
    print("\n" + ("ALL 7 FIGURES REGENERATED" if ok else "SOME FIGURES FAILED (see above)"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(run())
