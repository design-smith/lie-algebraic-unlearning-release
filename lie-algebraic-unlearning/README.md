# Lie-Algebraic Unlearning

Code and results for *Lie Algebraic Optimization for Continual Learning and Targeted Unlearning
in Neural Networks*. The method replaces additive fine-tuning with multiplicative orthogonal
updates `W_new = exp(A) W` (`A` skew-symmetric); sequences compose as products of exponentials
whose interference is carried by the commutators `[A_i, A_j]`.

## Install

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
```

## Reproduce every figure

```bash
python make_figures.py
```

Writes all seven paper figures to `figures/` (exact paper filenames), recomputing Figures 1(a-d)
and 2(a-b) from scratch on CPU and rebuilding Figure 3 from `results/experiment1_results.csv`
with its statistics recomputed from the same rows. Exits non-zero if anything fails.

| Figure | Script | What it shows |
| --- | --- | --- |
| 1(a-c) | `matrix_bch.py` | matrix-level BCH vs naive composition error |
| 1(d) | `truncation_sweep.py` | rank-truncated composition error vs the Theorem 1 bound |
| 2(a-b) | `toy_mlp.py` | MNIST MLP geometry preservation + forget-retain trade-off |
| 3 | `make_figures.py` (from CSV) | commutator vs sequential-unlearning interference on Llama-3.2-1B |

## Re-run the H1 experiment (GPU)

`h1_llama.py` is self-contained: it injects 80 synthetic facts into Llama-3.2-1B via LoRA,
unlearns them as four sequential Lie-rotation requests under entity- vs attribute-partitioning,
and correlates the normalized commutator with interference. A free Colab T4 suffices (~10-25 min):
set a Colab secret `HF_TOKEN` with access to `meta-llama/Llama-3.2-1B`, then `%run h1_llama.py`.
It writes `experiment1_results.csv` (drop into `results/`) plus a scatter. `SMOKE=1 python
h1_llama.py` validates the full code path on CPU in ~2 minutes without the gated download.

## License

MIT. See [LICENSE](LICENSE).
