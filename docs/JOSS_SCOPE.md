# GPCell and JOSS

## What the JOSS story should be

The dissertation is broad: it covers biological motivation, GP methodology, performance work, and exploratory Bayesian extensions.

A JOSS submission should be narrower and more software-centered.

The strongest current software narrative is:

- GPCell is an open-source Python framework for Gaussian-process-based oscillation detection in single-cell time series.
- It reproduces the core MATLAB-era oscillation-classification workflow in a modern Python stack.
- It adds a modular architecture for priors, kernels, inference modes, and batch fitting.
- It improves usability and runtime through parallel fitting, memmaps, tests, and documentation.

That is already a credible JOSS contribution if presented cleanly.

## What the repository already has

- An OSI-approved license.
- A real reusable Python package under `gpcell/`.
- `pyproject.toml` package metadata and dependency declarations.
- Tests tied to an included dataset.
- CI for linting and publishing generated API docs.
- An example dataset and simulation utilities.
- A functional API with typed result objects alongside the original `OscillatorDetector`
  workflow.
- A dissertation that already explains motivation, methods, and evaluation.

## What still needs work for a strong JOSS submission

### Essential

- Create JOSS paper files:
  - `paper.md`
  - `paper.bib`
- Keep install instructions polished and tested for a fresh user.
- Add citation metadata suitable for JOSS and archival releases.
- Tighten the README so it sells the software, not just the dissertation.
- Make at least one reproducible user example easy to run.

### Strongly recommended

- Add `CITATION.cff`.
- Add a contributor guide or expand `docs/DEVELOPMENT.md`.
- Separate maintained examples from archival notebooks.
- Clarify which methods are stable production workflows versus ongoing research extensions.
- Document benchmark conditions more explicitly.

### Nice to have

- Add a changelog.
- Add versioned releases.
- Publish example outputs or a small tutorial notebook in a cleaned form.

## Suggested JOSS positioning

### Summary sentence

GPCell is a Python library for Gaussian-process-based detection of oscillatory single-cell gene expression dynamics, with an emphasis on reproducibility, extensibility, and performant batch fitting.

### Primary audience

- computational biologists studying dynamic gene regulation,
- researchers reproducing or extending Phillips-style GP oscillation detection,
- developers needing a reusable GPflow-based implementation rather than a one-off notebook.

### Key software contributions to emphasize

- modular GP construction via `GPRConstructor`,
- reusable `GaussianProcess` wrapper over GPflow models,
- domain workflow API via `OscillatorDetector`,
- functional workflow API via `run_pipeline(...)` and typed dataclass results,
- parallel fitting for many traces and replicates,
- included simulation and regression-testing utilities.

## What to de-emphasize in the first JOSS submission

- ambitious claims around fully Bayesian classification unless the path is stabilized further,
- dissertation-only breadth,
- every benchmark or notebook experiment,
- unfinished bridge-sampling directions.

Those can still be mentioned as extensions, but they should not be the load-bearing novelty claim.

## Suggested near-term backlog

1. Add `CITATION.cff`.
2. Create `paper.md` and `paper.bib` from the software contribution, not by shrinking the
   dissertation wholesale.
3. Add one polished example script or notebook using `OscillatorDetector.from_file(...)` or
   the functional `run_pipeline(...)` API.
4. Decide what the supported workflows are:
   - likely `BIC`,
   - likely `BOOTSTRAP`,
   - maybe `MCMC` as experimental.
5. Tighten CI so the full documented test set runs in the documented environment.

## A practical paper outline

1. Summary
   GPCell provides a tested, extensible, Python implementation of GP-based oscillation detection for single-cell time series.
2. Statement of need
   Existing approaches were MATLAB-centered, less accessible, and harder to extend with modern GP tooling.
3. Functionality
   Describe loading, preprocessing, OU vs OUosc fitting, classification, and simulation helpers.
4. Key implementation choices
   Highlight GPflow, TensorFlow Probability, Joblib, memmaps, and typed abstractions.
5. Example use
   Show the Hes dataset workflow.
6. Related work
   Compare to the earlier MATLAB approach and general GP frameworks.

## Bottom line

The repo is already close enough in substance for JOSS. The main missing pieces are citation
metadata, JOSS paper files, sharper software framing, and a cleaner maintained example for a
new user to run and understand the stable features.
