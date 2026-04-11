# Development Guide

## Setup

The preferred local setup is `uv` with a project-local virtualenv. CI uses Python 3.12, so
use that unless you are deliberately testing another interpreter:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[ci]"
```

Why the setuptools pin exists:

- `gpflow` currently imports `pkg_resources`.
- that means the environment needs `setuptools<81` until the project or its dependencies move away from that import.

`environment.yml` is still present as a legacy Conda description, but `uv` is the
recommended path going forward.

## Common commands

Run the main behavior tests:

```bash
.venv/bin/python -m unittest tests.gpflow_tests
```

Run the slower correctness regression tests, which compare against the earlier notebook
workflow:

```bash
.venv/bin/python -m unittest tests.correctness
```

Run the lighter regression tests for data-shape behavior and the functional API:

```bash
.venv/bin/python -m unittest tests.characterization_tests tests.functional_api_tests
```

Run the full maintained test set explicitly:

```bash
.venv/bin/python -m unittest \
  tests.gpflow_tests \
  tests.correctness \
  tests.characterization_tests \
  tests.functional_api_tests
```

Run the Makefile target:

```bash
make all_tests
```

Note: the `Makefile` still assumes `python` exists on `PATH`, and `all_tests` only runs
`tests.gpflow_tests`. Direct `.venv/bin/python ...` commands are currently more complete and
more reliable.

Generate API docs:

```bash
.venv/bin/pdoc --docformat numpy gpcell
```

Or write them out statically:

```bash
.venv/bin/pdoc --docformat numpy -o docs gpcell
```

If you do not want generated HTML mixed with handwritten docs, use a different output directory.

Run the Ruff checks used by CI:

```bash
.venv/bin/ruff check --select E9,F63,F7,F82 .
.venv/bin/ruff format --check gpcell tests
```

Run the broader local lint check:

```bash
.venv/bin/ruff check .
```

## Paper and dissertation build

The dissertation is a Quarto book in `scripts/dissertation/`. It mixes Python chunks in
`methods.qmd`, R chunks in `results.qmd`, and static image assets.

Install the additional render dependencies if the environment does not already have them:

```bash
uv pip install --python .venv/bin/python nbformat nbclient nbconvert
Rscript -e 'install.packages(c("rmarkdown", "tidyverse"), repos="https://cloud.r-project.org")'
```

Render the PDF from the repository root:

```bash
PATH="$(pwd)/.venv/bin:$PATH" quarto render scripts/dissertation --to pdf
```

The `PATH` prefix tells Quarto to use the project virtualenv for Python execution. Without
it, Quarto may pick a system Python that does not have `yaml`, Jupyter, GPflow, or the
project dependencies installed.

After rendering, compare generated artifacts against the tracked baseline:

```bash
git status --short
git diff --stat
git show HEAD:scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf > /tmp/gpcell-head.pdf
pdftotext /tmp/gpcell-head.pdf /tmp/gpcell-head.txt
pdftotext scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf /tmp/gpcell-new.txt
diff -u /tmp/gpcell-head.txt /tmp/gpcell-new.txt
```

Small PDF binary differences can come from renderer metadata even when extracted text is
unchanged. Treat notebook output clearing and PDF metadata churn as generated-artifact
noise unless source cells or extracted content changed.

## Working model of the codebase

When changing behavior, think in terms of these layers:

1. `gpcell/main.py`
   This is the workflow orchestration layer and public entry point.
2. `gpcell/utils.py`
   This is where fitting, preprocessing, I/O, simulation, and performance-sensitive helpers live.
3. `gpcell/backend/`
   This is the GPflow integration layer: constructors, priors, GP wrappers, low-level helpers.
4. `tests/`
   These lock in expected behavior against the Hes example and earlier results.

## Important implementation details

### CSV conventions

`OscillatorDetector.from_file(...)` expects a single CSV containing:

- one time column,
- one or more cell trace columns,
- one or more background trace columns.

The current example is `data/hes/Hes1_example.csv`.

### Replicates matter

The BIC pipeline does not fit a single OU and OUosc model per cell. It fits multiple replicates and keeps the highest-posterior result. That replicate logic is part of the method, not just a numerical workaround.

### Performance depends on data shape

`fit_processes_joblib(...)` has different branches for:

- small non-homogeneous traces,
- homogeneous traces that can be stacked directly,
- larger non-homogeneous traces that are padded and memory-mapped.

If a performance change looks surprising, check which branch your dataset takes.

### Temporary files

Parallel fitting writes memmap data into `./temp`. This directory is operational, not archival.

### MCMC support is experimental

The MCMC path exists and is scientifically interesting, but the most stable and clearly validated path in the repo today is still the BIC/bootstrap workflow.

### Functional API status

The functional API in `gpcell/functional.py` now provides `DetectorConfig`,
`DetectorData`, `PreparedData`, `BICResult`, `BootstrapResult`, `PipelineResult`,
`load_detector_data(...)`, `prepare_detector_data(...)`, `fit_bic(...)`,
`fit_bootstrap(...)`, and `run_pipeline(...)`.

`OscillatorDetector` remains the compatibility API and delegates the BIC/bootstrap fitting
work to the functional layer. MCMC is still kept on the class adapter path.

## Suggested contributor workflow

1. Reproduce behavior on the Hes dataset first.
2. Change one layer at a time.
3. Re-run `tests.gpflow_tests` after any change to fitting or data loading.
4. Re-run `tests.correctness` after changes that may affect regression parity with earlier results.
5. Update handwritten docs when behavior changes.
6. Keep dissertation-era exploratory notebooks separate from maintained library behavior.

## Good first cleanup targets

- Separate handwritten docs from generated API docs more cleanly.
- Add a documented install path for users outside the dissertation environment.
- Promote one notebook or script into a polished end-to-end example.
- Add JOSS paper scaffolding: `paper.md`, `paper.bib`, and a clearer software-only contribution statement.

## Working with AI agents

For future AI-assisted work, the most useful context to provide or read first is:

- `README.md`
- `docs/ARCHITECTURE.md`
- `docs/JOSS_SCOPE.md`
- `AGENTS.md`

The quickest reliable way to reorient inside the codebase is:

1. inspect `gpcell/main.py`,
2. inspect `gpcell/utils.py`,
3. inspect `tests/gpflow_tests.py`,
4. only then branch into backend details or notebooks.

## Current environment caveat

If tests fail immediately with import errors such as missing `numpy`, the problem is usually the active `uv` environment rather than the project logic. Recreate or reactivate `.venv` before treating that as a code regression.

Apply safe automatic fixes:

```bash
.venv/bin/ruff check . --fix
```
