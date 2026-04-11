# GPCell Agent Notes

This file is for future AI or automation-assisted work in this repository. Keep it concise
and update the project docs whenever commands, workflows, or supported behavior change.

## Start Here

Read these first:

1. `README.md`
2. `docs/DEVELOPMENT.md`
3. `docs/ARCHITECTURE.md`
4. `docs/JOSS_SCOPE.md`
5. `gpcell/main.py`
6. `gpcell/functional.py`
7. `gpcell/utils.py`
8. `tests/gpflow_tests.py`
9. `tests/correctness.py`

## Project Shape

GPCell is a Python/GPflow framework for classifying single-cell gene expression traces as
oscillatory or non-oscillatory.

- Main compatibility API: `gpcell.OscillatorDetector`.
- Newer functional API: `gpcell.functional` via `DetectorConfig`, `DetectorData`,
  `fit_bic(...)`, `fit_bootstrap(...)`, and `run_pipeline(...)`.
- `main.py`: class adapter, public workflow, plotting compatibility.
- `functional.py`: dataclass-based BIC/bootstrap workflow used by the class adapter.
- `utils.py`: fitting, preprocessing, I/O, simulation, memmap, and performance logic.
- `backend/`: GPflow integration, priors, types, and lower-level helpers.
- `tests/`: Hes example behavior, regression parity, characterization, and functional API checks.
- `scripts/dissertation/`: Quarto dissertation source and rendered PDF. Treat it as important
  context, but do not assume every dissertation experiment is a supported library feature.

## Current Facts

- Preferred setup is `uv venv .venv --python 3.12` followed by
  `uv pip install --python .venv/bin/python -e ".[ci]"`.
- `pyproject.toml` now contains package metadata and the main dependency list.
- `gpflow` still imports `pkg_resources`, so keep `setuptools<81` via the `ci` extra unless
  dependency behavior changes.
- The BIC/bootstrap path is the grounded workflow; MCMC remains experimental and class-only.
- Parallel fitting writes temporary memmap files to `./temp`.
- Performance depends on whether traces are homogeneous in length.
- The `Makefile` still assumes `python` and `make all_tests` only runs `tests.gpflow_tests`;
  prefer explicit `.venv/bin/python -m unittest ...` commands.
- Quarto PDF rendering needs notebook execution packages in `.venv` and R `rmarkdown` /
  `tidyverse`; run Quarto with `PATH="$(pwd)/.venv/bin:$PATH"` so it uses the project env.
- Small PDF binary diffs can be renderer metadata; compare extracted text before treating them
  as scientific result changes.

## High-Value Commands

```bash
.venv/bin/python -m unittest tests.gpflow_tests
.venv/bin/python -m unittest tests.correctness
.venv/bin/python -m unittest tests.characterization_tests tests.functional_api_tests
.venv/bin/ruff check .
.venv/bin/ruff format --check gpcell tests
PATH="$(pwd)/.venv/bin:$PATH" quarto render scripts/dissertation --to pdf
```

## Working Rules

- Read code before proposing architecture changes.
- Preserve the Hes example workflow unless explicitly changing methodology.
- If touching fitting behavior, inspect and run the relevant tests, especially
  `tests.gpflow_tests.py` and `tests.correctness.py`.
- If touching docs, update `README.md`, `docs/DEVELOPMENT.md`, and other relevant docs in the
  same change. Do not leave run commands stale.
- If making JOSS-facing changes, keep the software contribution tighter than the dissertation
  narrative and keep `docs/JOSS_SCOPE.md` current.
- Do not treat ignored docs or generated notebooks/PDF metadata churn as source changes without
  checking whether tracked content or extracted output actually changed.
