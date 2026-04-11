## Testing directory for the library

Contains unit tests to deal with both functionality and performance checks. The tests are written using the `unittest` framework and can be run using the `unittest` module.

Run the maintained test modules from the repository root:

```bash
.venv/bin/python -m unittest \
  tests.gpflow_tests \
  tests.correctness \
  tests.characterization_tests \
  tests.functional_api_tests
```

Useful smaller runs:

```bash
.venv/bin/python -m unittest tests.gpflow_tests
.venv/bin/python -m unittest tests.correctness
.venv/bin/python -m unittest tests.characterization_tests tests.functional_api_tests
```

`tests.correctness` is slower and checks regression parity with the earlier notebook-style
analysis. `tests.gpflow_tests` covers the main `OscillatorDetector` behavior, including the
Hes example BIC count and joblib/bootstrap branches. The characterization and functional
API tests are lighter checks for data-shape assumptions and typed pipeline results.
