# GPCell Agent Notes

This file is for future AI or automation-assisted work in this repository.

## Start here

Read these first:

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/JOSS_SCOPE.md`
4. `gpcell/main.py`
5. `gpcell/utils.py`
6. `tests/gpflow_tests.py`

## What the project is

GPCell is a Python/GPflow framework for classifying single-cell gene expression traces as oscillatory or non-oscillatory.

The main public object is `gpcell.OscillatorDetector`.

The repository also contains dissertation material in `scripts/dissertation/`. Treat that as important context, but do not assume every dissertation experiment is a stable supported library feature.

## Stable mental model

- `main.py`: orchestration and public workflow.
- `utils.py`: fitting utilities, preprocessing, I/O, simulation, and performance logic.
- `backend/`: GPflow integration, priors, types, lower-level helpers.
- `tests/`: expected behavior on the Hes example and regression checks.

## High-value entry points

- `OscillatorDetector.from_file(...)`
- `OscillatorDetector.fit(...)`
- `utils.fit_processes(...)`
- `utils.fit_processes_joblib(...)`
- `utils.detrend(...)`
- `utils.background_noise(...)`

## Important caveats

- Parallel fitting writes temporary memmap files to `./temp`.
- Performance behavior depends on whether traces are homogeneous in length.
- The BIC/bootstrap path is the most grounded workflow.
- The MCMC path exists but is still closer to research exploration.
- The `Makefile` assumes `python`; some local shells may only have `python3`.
- The preferred local environment is now a `uv`-managed `.venv`.
- `gpflow` currently needs `setuptools<81` in that environment because it still imports `pkg_resources`.

## Good default behavior for future agents

- Prefer reading code before proposing architecture changes.
- Preserve the Hes example workflow unless explicitly changing methodology.
- If touching fitting behavior, inspect `tests/gpflow_tests.py` and `tests/correctness.py`.
- If making JOSS-facing changes, keep the software contribution tighter than the dissertation narrative.
