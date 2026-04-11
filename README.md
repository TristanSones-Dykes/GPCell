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

## Quick start

The recommended local workflow is now `uv`.

Create and populate a virtual environment:

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python "setuptools<81"
uv pip install --python .venv/bin/python \
  numpy matplotlib pandas pdoc ipykernel tensorflow tensorflow-probability \
  gpflow tf-keras numba joblib astropy ruff pymc scipy arviz scikit-learn
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

## Repository map

- `gpcell/`: main library code.
- `gpcell/backend/`: GP constructors, priors, types, parameter assignment helpers, simulation kernels.
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

Generated API docs are published from CI with `pdoc`. Locally, the historical commands are:

- `pdoc gpcell`
- `pdoc -o docs gpcell`

If you want to keep the handwritten docs and generated API docs separate, prefer a local output directory such as `site/` or `docs/api/`.

## Environment and testing

Typical `uv` commands:

- `uv venv .venv --python 3.10`
- `uv pip install --python .venv/bin/python "setuptools<81"`
- `uv pip install --python .venv/bin/python numpy matplotlib pandas pdoc ipykernel tensorflow tensorflow-probability gpflow tf-keras numba joblib astropy ruff pymc scipy arviz scikit-learn`
- `.venv/bin/python -m unittest tests.gpflow_tests`
- `.venv/bin/python -m unittest tests.correctness`

Notes:

- `gpflow` still imports `pkg_resources`, so the environment currently needs `setuptools<81`.
- `environment.yml` remains in the repository as a legacy Conda description, but `uv` is now the preferred local setup.
- The current `Makefile` still assumes a `python` executable is available on `PATH`.

## Dissertation to JOSS

The dissertation is the best source for the scientific background and evaluation narrative, but a JOSS paper will need a tighter software contribution story. The working notes for that transition are in:

- [`docs/JOSS_SCOPE.md`](docs/JOSS_SCOPE.md)
