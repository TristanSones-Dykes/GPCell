# GPCell Architecture

## Project scope

GPCell is not just a collection of GP utilities. It currently embodies a specific scientific workflow for detecting oscillatory single-cell gene expression traces, especially the Hes1-style analysis used in the dissertation.

At a high level, the repository has two layers:

- a reusable GP-fitting layer built on GPflow and TensorFlow Probability,
- a domain workflow layer exposed through `OscillatorDetector`.

The dissertation source in `scripts/dissertation/` is important context, because several design choices in the codebase are easiest to understand through that biological and statistical motivation.

## End-to-end pipeline

The main workflow in [`gpcell/main.py`](../gpcell/main.py) is:

1. `OscillatorDetector.from_file(...)` reads a CSV and splits columns into time, cell traces, and background traces.
2. `background_noise(...)` fits RBF GPs to background traces and estimates a shared noise level.
3. `detrend(...)` standardizes each cell trace and removes slow trends with an RBF GP.
4. `_fit_ou_ouosc(...)` fits replicate GP models for:
   - OU-like non-oscillatory behavior via `gpflow.kernels.Matern12`,
   - oscillatory behavior via `[Matern12, Cosine]`, combined multiplicatively by `GPRConstructor`.
5. `_calc_gpr_bic_llr(...)` selects the best replicate per model family and computes:
   - log-likelihood ratios,
   - BIC differences,
   - a rough period estimate from the oscillatory covariance.
6. Optional follow-on analyses:
   - `_fit_bootstrap(...)` simulates synthetic non-oscillatory traces and derives q-value style thresholds,
   - `_fit_mcmc(...)` samples oscillatory model parameters and experiments with Savage-Dickey Bayes factors.

## Main modules

### `gpcell/main.py`

Home of `OscillatorDetector`, the main interface for loading data, preprocessing, fitting, classification, and plot generation.

Important behavior:

- `fit(methods=...)` accepts `"BIC"`, `"BOOTSTRAP"`, and `"MCMC"`.
- bootstrap depends on BIC results.
- plot requests are configured at construction time through the `plots` parameter.
- `joblib=True` switches the OU/OUosc fitting path to the parallel utilities.

### `gpcell/utils.py`

General utility layer for:

- fitting many GP models,
- parallel fitting with Joblib,
- memmap/padding strategies for large or homogeneous trace sets,
- detrending and background-noise estimation,
- CSV loading helpers,
- Gillespie-based simulation utilities for reproducing benchmark datasets.

This file is operationally important. A large part of GPCell's performance story lives here rather than in the high-level API.

### `gpcell/backend/gpr_constructor.py`

`GPRConstructor` turns a kernel specification, prior factory, trainable-parameter flags, and inference mode into a concrete GPflow model.

This separation matters because it lets GPCell:

- create many similar models cleanly,
- swap between exact regression and MCMC-backed `GPMC`,
- assign priors and trainable flags dynamically.

### `gpcell/backend/gaussian_process.py`

`GaussianProcess` is a small wrapper around GPflow models that standardizes:

- fitting,
- posterior prediction,
- log-posterior extraction,
- optional MCMC execution,
- a few plotting and diagnostics helpers.

In practice, this is the lowest-level object most of the rest of the package works with.

### `gpcell/backend/priors.py`

Contains the built-in prior generators and trainability presets for:

- Hes-data ML-style initialization,
- simulated-data initializations,
- MCMC-specific priors used in dissertation experiments.

These priors are a key part of the scientific story, not just implementation detail.

## Data model assumptions

The main CSV loader in `load_data(...)` expects:

- one time column,
- multiple trace columns sharing a prefix such as `Cell`,
- multiple background columns sharing a prefix such as `Background`.

The included Hes example follows this pattern:

- `Time (h)`
- `Cell 1`, `Cell 2`, ...
- `Background`, repeated several times

Trailing zeros are treated as missing tail values and trimmed per trace.

## Performance strategy

The repository's performance claims come from a combination of:

- Joblib parallelism for per-trace GP fitting,
- NumPy memmaps for homogeneous or padded trace matrices,
- Numba-compiled Gillespie simulators,
- reusable constructors instead of rebuilding logic inside every worker.

This is why `utils.fit_processes_joblib(...)` and the backend worker functions are central to the project.

## What is already solid

- There is a clear core API.
- The Hes example is wired into tests.
- The dissertation sources explain the intended methods and contribution.
- CI exists for linting and publishing API docs.
- The codebase already contains regression-minded tests, not just smoke tests.

## What still feels research-grade

- Packaging is still repo-centric rather than polished for installation from PyPI.
- Notebook and script organization is mixed between archival and active material.
- The MCMC/Bayes-factor path is still experimental relative to the BIC workflow.
- Handwritten project docs were previously minimal.
- JOSS-facing metadata and paper scaffolding are not yet in place.

## Recommended reading order

If you are new to the repo, read in this order:

1. [`README.md`](../README.md)
2. [`docs/JOSS_SCOPE.md`](JOSS_SCOPE.md)
3. [`gpcell/main.py`](../gpcell/main.py)
4. [`gpcell/utils.py`](../gpcell/utils.py)
5. [`gpcell/backend/gpr_constructor.py`](../gpcell/backend/gpr_constructor.py)
6. [`gpcell/backend/gaussian_process.py`](../gpcell/backend/gaussian_process.py)
7. [`tests/gpflow_tests.py`](../tests/gpflow_tests.py)
8. dissertation chapters in `scripts/dissertation/`

