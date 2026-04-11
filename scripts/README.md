## Scripts folder

This folder contains dissertation-era notebooks, paper figure work, and small exploratory
scripts. Treat most notebooks here as archival or development material unless a task
explicitly asks for them.

- `figures.ipynb`: notebook used for paper/dissertation figure work. It includes expensive
  GP fitting and MCMC cells, so do not assume it is a quick smoke test.
- `ROC.ipynb`: exploratory ROC-generation work.
- `MCMC.ipynb`: exploratory MCMC workflow.
- `tutorial_verification/`: tutorial notebooks and verification drafts.
- `dissertation/`: Quarto dissertation source, static assets, and rendered PDF.
- `test.py`: small multiprocessing experiment, not the main project test runner.

Run maintained tests from the repository root instead:

```bash
.venv/bin/python -m unittest \
  tests.gpflow_tests \
  tests.correctness \
  tests.characterization_tests \
  tests.functional_api_tests
```

Render the dissertation PDF from the repository root with:

```bash
PATH="$(pwd)/.venv/bin:$PATH" quarto render scripts/dissertation --to pdf
```
