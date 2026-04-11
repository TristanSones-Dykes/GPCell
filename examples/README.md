# Examples

This directory contains maintained examples for the reusable GPCell package. The older
notebook reconstructions under `scripts/` are archival context rather than the first place
new users should start.

Run the canonical Hes1 BIC workflow from the repository root:

```bash
.venv/bin/python examples/hes1_bic_workflow.py
```

For a faster smoke run while checking an environment:

```bash
.venv/bin/python examples/hes1_bic_workflow.py --max-cells 3 --bic-replicates 1
```

The default example uses the functional API (`load_detector_data(...)` and
`run_pipeline(...)`). It focuses on the stable BIC workflow, with bootstrap available as an
explicit opt-in:

```bash
.venv/bin/python examples/hes1_bic_workflow.py --bootstrap --joblib
```
