# Technique Validation Results

Date: 2026-04-11

This note summarizes the test and pipeline validation run after adding the Phase 2 enhanced
kernel/LMC and SVGP work. The validation environment was the project `.venv` currently present
in this checkout. It reports Python 3.10.19, even though the preferred documented setup is
Python 3.12.

## Test Results

The focused Phase 2 pytest suite passed:

```bash
.venv/bin/python -m pytest tests/enhanced_kernel_tests.py tests/svgp_tests.py
```

Result: 5 passed.

The lighter functional and characterization tests passed:

```bash
.venv/bin/python -m unittest tests.characterization_tests tests.functional_api_tests
```

Result: 8 passed. TensorFlow emitted expected numerical warnings during a few GP fits where
intermediate Cholesky decompositions produced NaNs; the tests still completed successfully.

The correctness parity suite passed:

```bash
.venv/bin/python -m unittest tests.correctness
```

Result: 6 passed. The notebook-parity noise estimate remained `7.3022215754897335`, and the
Hes example BIC classification remained 10/12 oscillatory cells.

The full GPflow/Hes behavior suite was run twice:

```bash
.venv/bin/python -m unittest tests.gpflow_tests
```

The first run failed one stochastic joblib bootstrap classification assertion. A clean rerun
passed: 8 passed. The BIC portions of both runs passed. Treat this as evidence that the
bootstrap test still has random sensitivity rather than as a deterministic regression.

Ruff results:

```bash
.venv/bin/ruff format --check gpcell tests
```

Result: passed after formatting `gpcell/backend/gpr_constructor.py`.

```bash
.venv/bin/ruff check gpcell/backend/gpr_constructor.py \
  gpcell/backend/gaussian_process.py \
  gpcell/backend/kernels.py \
  tests/enhanced_kernel_tests.py \
  tests/svgp_tests.py
```

Result: passed for the touched implementation and test files.

```bash
.venv/bin/ruff check .
```

Result: failed due to broader pre-existing lint issues across older modules, scripts, notebooks,
and tests. The failures include import ordering, unused imports, `zip(..., strict=...)` guidance,
notebook lint, and mutable default warnings in files outside this task.

## Pipeline Results

### BIC Workflow

Command:

```bash
.venv/bin/python examples/hes1_bic_workflow.py --bic-replicates 10
```

Result: passed on all 12 bundled Hes1 traces. The estimated background noise was `7.302`, and
the BIC cutoff of 3 classified 10/12 traces as oscillatory.

Intuition: this is the grounded GPCell workflow. It fits OU and OU x Cosine models and rewards
the oscillatory model only when its likelihood gain is large enough to overcome the BIC penalty
for the extra oscillatory parameter. The two traces below the cutoff still had some structure,
but not enough to justify the richer oscillatory covariance under this criterion.

### Bootstrap Workflow

Command:

```bash
.venv/bin/python examples/hes1_bic_workflow.py \
  --max-cells 3 \
  --bic-replicates 2 \
  --bootstrap \
  --bootstrap-replicates 10
```

Result: passed as a smoke workflow. The first three Hes1 traces were classified 3/3
oscillatory by the BIC cutoff and 3/3 oscillatory by bootstrap q-value filtering.

Intuition: bootstrap wraps the BIC/LLR machinery with synthetic non-oscillatory traces. It asks
whether a real trace is more oscillatory than traces sampled from the fitted non-oscillatory
background/trend structure. This makes the result easier to interpret as a thresholded screen,
but the workflow is stochastic and much more expensive because each synthetic trace is itself
refit with OU and OU x Cosine models.

### Time-Varying Cosine Kernel

A direct exact-GPR smoke fit was run on a synthetic chirp-like trace using
`TimeVaryingCosine`. The fitted parameters were:

- `beta_0`: `0.44999999100040133`
- `beta_1`: `0.07999999706310305`
- log posterior: `174.7473769858277`
- prediction shapes: mean `(5, 1)`, variance `(5, 1)`

Intuition: the new kernel replaces a fixed-frequency cosine phase with an integrated phase
`u(x) = beta_0*x + 0.5*beta_1*x^2`. In the synthetic trace, the optimizer recovered the intended
base frequency and frequency drift closely. This is the right behavior for chirp-like biological
signals where a fixed-period model would smear the covariance over time.

### LMC/Coregion Multivariate GP

A two-output synthetic trace was compiled through `GPRConstructor(..., coregion_output_dim=2)`.
The model had finite training loss `17.959655031089177`, augmented input shape `(64, 2)`, and
flattened target shape `(64, 1)`.

Intuition: the LMC path converts independent-looking per-output observations into one GPflow
model with an added output-index column. The temporal kernel acts on time, while
`gpflow.kernels.Coregion` learns output-to-output covariance. This is the entry point for
coupled gene traces where shared timing, lagged structure, or correlated amplitudes matter.

### SVGP Approximate Inference

A sparse variational GP smoke fit used KMeans inducing points and Adam updates. Results:

- initial minibatch loss: `2152.281111111229`
- final minibatch loss: `7.6670418298225735`
- full-data ELBO: `-11.169848832218726`
- inducing point shape: `(8, 1)`
- prediction shapes: mean `(5, 1)`, variance `(5, 1)`

Intuition: SVGP avoids exact GP matrix inversion over all observations by summarizing the
process through inducing points. In this smoke run, KMeans placed eight inducing points over
the input domain and Adam rapidly reduced the variational objective. The tradeoff is that the
reported objective is an ELBO approximation rather than the exact GPR log posterior.

### Experimental MCMC Backend

A deliberately tiny backend smoke sampled an OU x Cosine GPMC model with 8 burn-in steps and 8
samples, using the repository's existing MCMC prior setup. It produced four sample blocks, with
the first block shape `(8, 8, 1)`, and tracked these parameters:

- `.V`
- `.kernel.kernels[0].lengthscales`
- `.kernel.kernels[0].variance`
- `.kernel.kernels[1].lengthscales`

Intuition: this validates that the experimental GPflow/TFP sampling path still constructs and
samples when parameters have priors. It is not the same as a full scientific MCMC run. The
class-level MCMC workflow remains expensive because it is hard-coded to multiple chains and
large sampler defaults, so full MCMC validation should be treated as a separate planned run.

## Overall Interpretation

The stable BIC workflow remains intact and reproduces the expected Hes behavior. Bootstrap also
works, but its stochastic thresholding makes isolated classification assertions fragile unless
the random state and parallel execution are tightly controlled. The Phase 2 additions compile
and run in focused pipelines: TimeVaryingCosine handles non-stationary frequency, LMC/coregion
handles multivariate outputs through output indexing, and SVGP gives a working approximate
inference path with clear loss reduction. MCMC remains experimental and should not be conflated
with the maintained BIC/bootstrap path until its runtime and sampling controls are made more
explicit.
