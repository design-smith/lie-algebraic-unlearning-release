# Submission packet

Three buckets. Everything a stranger needs to clone, install, and regenerate every figure, plus
the arXiv source and the workshop stub.

```
submission_packet/
├── lie-algebraic-unlearning/     BUCKET 1 - the public repo ([repo URL] points here)
│   ├── README.md  LICENSE  requirements.txt  .gitignore
│   ├── matrix_bch.py            Exp 1a-c
│   ├── truncation_sweep.py      Exp 1d
│   ├── toy_mlp.py               Exp 2a-b
│   ├── h1_llama.py              the H1 experiment (GPU; SMOKE=1 runs on CPU)
│   ├── make_figures.py          reads results/ -> writes all 7 PNGs to figures/
│   ├── results/experiment1_results.csv
│   └── figures/Exp1_1..Exp1_4, Exp2_1, Exp2_2, Exp_H1  (.png)
├── arxiv/                        BUCKET 2 - arXiv source (also zipped as arxiv.tar.gz)
│   ├── paper.tex                 pandoc-converted + hand-cleaned
│   ├── references.bib            42 refs, BibTeX
│   └── Exp1_1..Exp_H1 (.png)     the 7 figures
├── arxiv.tar.gz                  the arXiv upload (flat: paper.tex + references.bib + 7 PNGs)
└── workshop/
    └── workshop.tex              BUCKET 3 - set-aside stub (built last on the venue template)
```

## Status

- **Bucket 1 verified.** `h1_llama.py` was consolidated from the Colab session and SMOKE-run
  end to end (CPU, tiny model) confirming it regenerates `experiment1_results.csv`; then
  `make_figures.py` regenerated **all 7 PNGs** including `Exp_H1.png` rebuilt from that CSV
  (rho=0.250, matches the paper). The bar "clone, install, regenerate every figure with one
  command each" is met.
- **Bucket 2 assembled, not compiled.** No LaTeX toolchain on the build machine, so `paper.tex`
  was verified structurally only (balanced environments: document/abstract/7 figures/3 longtables/
  5 enumerate all balanced; title+abstract+heading hierarchy rebuilt from the pandoc output;
  redundant figure captions removed; braces balanced; UTF-8 clean). You compile it in step 3 below.
- **Bucket 3 is a stub by design** - the workshop cut goes on the venue's own template, which does
  not exist until you pick the venue.

## paper.tex notes (read before compiling)

- Compiles with **pdflatex** in one pass, **no bibtex needed**: the References section is a
  formatted list (from the manuscript), and body citations are author-year text. `references.bib`
  ships alongside as the machine-readable version (reuse it for the workshop cut or switch to
  `\bibliography{references}` + `\citep` later if a venue requires it).
- Author line is `Nathan Gandawa \\ nathan.gandawa@techtorch.io` - edit affiliation as needed.
- Section symbols (U+00A7) render via `inputenc utf8`; the preamble pandoc emitted is complete
  (amsmath, amssymb, graphicx, booktabs, longtable, hyperref, bookmark).
- Known safe-but-improvable: the 7 figures use the manuscript's bold "Figure N. ..." caption
  paragraphs (self-labeled, matching the source) rather than LaTeX float `\caption{}`, so they are
  not auto-numbered floats. Body references figures by name ("Figure 3"), so nothing breaks; if a
  venue wants numbered floats, move each bold paragraph into a `\caption{}` inside its `figure` env.

## Next steps (your plan; not done here - each is yours to drive)

1. **DONE** - `h1_llama.py` consolidated and verified (the only real-risk step).
2. **Repo up, public.** The five Section-A manuscript edits (conclusion, 6.1 lead-out, n=144/113,
   orderings note, rho-magnitude sentence) and both `[repo URL]` placeholders are **already in the
   manuscript** and flowed into `paper.tex`. The live repo is currently PRIVATE at
   github.com/design-smith/lie-algebraic-unlearning; flip it public when ready (bucket 1 here is
   the clean flat layout to publish).
3. **Compile.** `cd arxiv && pdflatex paper.tex` (twice for refs/labels), read the PDF end to end.
4. **arXiv.** Upload `arxiv.tar.gz`. **Prerequisite (longest lead-time, start now):** request a
   **cs.LG endorsement** so the upload is not blocked.
5. **Workshop.** Build `workshop/workshop.tex` on the chosen venue's template from the compiled
   paper (the stub lists exactly what to keep and cut).

## Before you publish (two must-dos I could not close from here)

1. **Add `experiment1_perlayer.csv` to `lie-algebraic-unlearning/results/`.** The paper cites the
   per-layer breakdown ("distributed rather than localized... layers 13-14 rho=0.33"), and
   `h1_llama.py` writes this file, but it is your Colab run's output and I do not have it. Releasing
   the repo without it reads as withheld data. Drop in the 2304-row CSV from your run.
2. **If you flip the existing private repo public instead of publishing this clean bucket,** strip
   the artifacts that must not ride along first: the old-title manuscript `.md`, the
   `Lie_Group_Orthogonal_Pipeline.ipynb` notebook, and the GPT-2 `Exp3_*.png` figures (not in the
   paper). This `submission_packet/lie-algebraic-unlearning/` bucket is already clean, so publishing
   it directly avoids the issue.

Note: the released `results/experiment1_results.csv` supports a gap-stratified reanalysis (adjacent
pairs null, multi-step windows rho~0.40), which the paper now reports in Section 6.2, so a reviewer
running that analysis finds the paper already there.

Explicitly not included (out of scope by request): Docker, CI, a PyPI package, a docs site, a
datasheet, custom `.bst`.
