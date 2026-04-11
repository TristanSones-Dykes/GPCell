# GPCell

GPCell is a Python framework for fitting Gaussian process models to single-cell gene expression time series and classifying traces as oscillatory or non-oscillatory.

This repository started as an MMath dissertation project and now contains three closely related things:

- the reusable `gpcell` library,
- tests and scripts used to validate the implementation against earlier MATLAB work,
- the dissertation sources and rendered PDF in `scripts/dissertation/`.

The current user-facing API is centered on `gpcell.OscillatorDetector`, which wraps the end-to-end Hes1-style workflow:

1. load time series from CSV,
2. estimate background noise from background traces,
3. detrend each cell trace with an RBF GP,
4. fit non-oscillatory OU and oscillatory OU x Cosine GP models,
5. classify traces with BIC, bootstrap-derived LLR thresholds, or MCMC-based Bayes-factor experiments.

## Why this project exists

The codebase is a Python and GPflow-based reworking of the Gaussian-process oscillation-detection approach from Phillips et al. It aims to keep the original scientific behavior while improving:

- openness and reproducibility,
- extensibility of priors, kernels, and inference,
- runtime through Joblib parallelism, memmaps, and Numba-backed simulation utilities,
- maintainability through typing, tests, and generated API docs.

The dissertation PDF is here:

- [`scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf`](scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf)

## Quick Start

The recommended local workflow is `uv` with a project-local virtual environment. CI uses
Python 3.12, so that is the best default for a fresh setup.

Create and populate a virtual environment:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e ".[ci]"
```

Examples below assume the project dependencies are installed in that environment.

```python
import gpcell as gc

detector = gc.OscillatorDetector.from_file(
    path="data/hes/Hes1_example.csv",
    X_name="Time (h)",
    background_name="Background",
    Y_name="Cell",
    params={
        "verbose": True,
        "joblib": True,
    },
)

detector.fit(methods="BIC")

print(detector.BIC_diffs)
print(sum(x > 3.0 for x in detector.BIC_diffs))
```

Common fitting modes:

- `BIC`: fits OU and OU x Cosine models and compares them with a BIC difference.
- `BOOTSTRAP`: runs the BIC pipeline, then generates synthetic non-oscillatory traces for thresholding.
- `MCMC`: fits oscillatory models with GPflow/TFP samplers and computes a Savage-Dickey-style Bayes-factor approximation.

## Functional API

GPCell exposes a functional pipeline in addition to `OscillatorDetector`. The class API
remains supported for existing scripts, while new workflow code can use typed dataclass
results:

```python
from gpcell import DetectorConfig, load_detector_data, run_pipeline

data = load_detector_data(
    path="data/hes/Hes1_example.csv",
    x_name="Time (h)",
    background_name="Background",
    y_name="Cell",
)
result = run_pipeline(data, DetectorConfig(joblib=True), methods=["BIC"])
print(result.bic.BIC_diffs)
```

The functional API currently supports the BIC and bootstrap paths. The MCMC path remains
available through `OscillatorDetector` and is still treated as experimental.

## Examples

Maintained examples live in [`examples/`](examples/). The canonical Hes1 functional API
workflow can be run from the repository root:

```bash
.venv/bin/python examples/hes1_bic_workflow.py
```

For a faster environment check, reduce the cell count and replicate count:

```bash
.venv/bin/python examples/hes1_bic_workflow.py --max-cells 3 --bic-replicates 1
```

## Repository map

- `gpcell/`: main library code.
- `gpcell/backend/`: GP constructors, priors, types, parameter assignment helpers, simulation kernels.
- `examples/`: maintained user-facing examples.
- `tests/`: regression and behavior tests against the Hes example and earlier implementations.
- `data/`: Hes example data plus simulated and MATLAB-derived benchmark datasets.
- `scripts/`: exploratory notebooks and supporting scripts.
- `scripts/dissertation/`: Quarto dissertation sources, figures, presentation assets, and rendered PDF.
- `docs/`: project-facing documentation added for ongoing maintenance and JOSS preparation.
- `.github/workflows/`: CI for linting and publishing generated API docs.

## Documentation

Start here for project-level docs:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)
- [`docs/JOSS_SCOPE.md`](docs/JOSS_SCOPE.md)
- [`AGENTS.md`](AGENTS.md)

Generated API docs are published from CI with `pdoc`. Locally, run:

- `.venv/bin/pdoc --docformat numpy gpcell`
- `.venv/bin/pdoc --docformat numpy -o docs gpcell`

If you want to keep the handwritten docs and generated API docs separate, prefer a local output directory such as `site/` or `docs/api/`.

## Environment and testing

Typical `uv` commands:

- `uv venv .venv --python 3.12`
- `uv pip install --python .venv/bin/python -e ".[ci]"`
- `.venv/bin/python -m unittest tests.gpflow_tests`
- `.venv/bin/python -m unittest tests.correctness`
- `.venv/bin/python -m unittest tests.characterization_tests tests.functional_api_tests`
- `.venv/bin/ruff check .`
- `.venv/bin/ruff format --check gpcell tests`

Notes:

- `gpflow` still imports `pkg_resources`, so the environment currently needs `setuptools<81`.
- `environment.yml` remains in the repository as a legacy Conda description, but `uv` is now the preferred local setup.
- The current `Makefile` still assumes a `python` executable is available on `PATH`.

## Paper and dissertation build

The rendered dissertation PDF lives at:

- [`scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf`](scripts/dissertation/_book/GPCell--A-Performant-Framework-for-Gaussian-Processes-in-Bioinformatics.pdf)

To rebuild it locally, install the extra notebook and R render dependencies:

```bash
uv pip install --python .venv/bin/python nbformat nbclient nbconvert
Rscript -e 'install.packages(c("rmarkdown", "tidyverse"), repos="https://cloud.r-project.org")'
PATH="$(pwd)/.venv/bin:$PATH" quarto render scripts/dissertation --to pdf
```

The `PATH` prefix matters because Quarto otherwise may choose a system Python without the
project's GPflow/Jupyter dependencies.

## Dissertation to JOSS

The dissertation is the best source for the scientific background and evaluation narrative, but a JOSS paper will need a tighter software contribution story. The working notes for that transition are in:

- [`docs/JOSS_SCOPE.md`](docs/JOSS_SCOPE.md)
