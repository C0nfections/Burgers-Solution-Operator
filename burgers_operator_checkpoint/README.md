# ESE 4020 Topic 12 - Burgers Solution Operator Baseline Checkpoint

This is a reproducible starting point the project.
It generates an independent-function dataset for the 1-D viscous Burgers PDE,
validates the numerical solver, and tunes principal component regression (PCR)
and ridge-regularized PCR on **validation data only**.

## Run it

Use Python 3.10+ with NumPy, pandas, scikit-learn, and matplotlib installed:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python operator_baselines.py all
```

The generator refuses to overwrite saved data by default. To deliberately
regenerate from the same seed, run `python operator_baselines.py generate
--overwrite`. To rerun just the model search, use
`python operator_baselines.py baselines`.

## What is a training pair?

Each independent initial function `u(x,0)` is sampled at 64 equally spaced
locations and paired with its solution `u(x,0.25)` sampled at the same points.
Consequently `X` and `Y` each have 64 columns, and 1,200 examples are split
into 800 training, 200 validation, and 200 held-out test **functions**.

The PDE and experimental parameters are:

- `u_t + (u^2 / 2)_x = 0.01 u_xx` on periodic `x in [0,1)`.
- Initial conditions are
  `0.25 * sum_{k=1}^5 (a_k*sin(2*pi*k*x) + b_k*cos(2*pi*k*x))/k`,
  with independent standard-normal `a_k,b_k`.
- All functions are generated with seed `4020`. The spatial solver has
  256 grid cells and conservative Rusanov flux for nonlinear advection.
- The output is sampled down to 64 points. Twenty-four additional initial
  functions are solved at 128, 256, and 512 cells to check grid convergence.

## Why two PCA ranks?

Input functions come from five sine and five cosine coefficients. They therefore
have at most ten independent directions even though each is stored as a
64-dimensional vector. The input PCA rank is searched over `2,4,6,8,10`.

The **nonlinear** Burgers evolution can generate more spatial frequencies.
The output rank is therefore tuned independently over `4,8,12,16,24,32`.
For each combination, PCR learns a multi-output linear map from input PCA
scores to output PCA scores. Ridge-PCR uses the same map with a squared-weight
penalty selected from `1e-6,1e-4,1e-2,1,100`.

Every PCA is fit on that repetition's **training subset only**. Both methods
receive exactly the same nested sample subsets for five fixed seeds. Model
selection uses mean per-function relative L2 error on the validation set.

## Data and outputs

| Path | Contents |
| --- | --- |
| `data/train.npz` | 800 training input/output functions, Fourier coefficients, IDs, grid |
| `data/validation.npz` | 200 validation functions, disjoint from training |
| `data/test_locked.npz` | 200 untouched test functions - do not inspect while tuning |
| `data/manifest.json` | Reproduction parameters, shapes, PDE sanity checks |
| `results/numerical_validation.json` | Resolution check for 24 separate functions |
| `results/best_validation_models.csv` | Selected rank, ridge penalty, and error for every N/seed |
| `results/validation_summary.csv` | Mean and standard deviation across seeds |
| `results/full_rank_scan_n800_seed4020.csv` | Full validation rank/penalty scan |
| `results/baseline_checkpoint.md` | Concise numerical results and interpretation |
| `results/figures/` | Grid convergence, learning curves, rank sensitivity, examples |

Load an NPZ file with
`with np.load("data/train.npz") as data: X, Y = data["X"], data["Y"]`.

## Evaluation rules

1. Do not read `test_locked.npz` while tuning PCA ranks, ridge penalties,
   DeepONet architectures, or other choices.
2. Do not fit a PCA, scaler, or other learned preprocessing on validation or
   test functions; never split grid points from the same function across sets.
3. Validation results in this checkpoint are **exploratory** because the same
   validation set selects hyperparameters. The plot shows mean +/- **one
   standard deviation across five training subsets**, not 95% confidence
   intervals. At N=800, the five seeds all use the *same* full training pool,
   so the seed SD is zero by construction; this does not indicate certainty.
4. After every design decision is frozen, evaluate on the test functions once
   and use a paired bootstrap over independent test functions for 95% intervals
   on DeepONet minus ridge-PCR error.
5. A small, negative validation error difference is not proof that one model
   generalizes better. A negative result (PCR performs as well or better) is
   still scientifically useful.

The full numerical implementation is in `operator_baselines.py`, organized
into solver, data generation, PCA regression, and plotting functions.
