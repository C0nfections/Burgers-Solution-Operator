#!/usr/bin/env python3
"""Generate a 1-D viscous Burgers benchmark and evaluate PCR/ridge-PCR.

The held-out test set is written to a separate file and is never loaded
by the baseline analysis. All numbers in the baseline report are validation
results, not final test results.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


@dataclass(frozen=True)
class SimulationConfig:
    n_pairs: int = 1200
    n_train: int = 800
    n_validation: int = 200
    n_test: int = 200
    n_solver: int = 256
    n_sensors: int = 64
    viscosity: float = 0.01
    final_time: float = 0.25
    n_fourier_modes: int = 5
    amplitude: float = 0.25
    seed: int = 4020
    cfl: float = 0.25


def make_initial_conditions(
    a: np.ndarray, b: np.ndarray, n_grid: int, amplitude: float
) -> np.ndarray:
    """Evaluate random Fourier-series initial conditions on a periodic grid."""
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("a and b must have matching (n_functions, n_modes) shapes")
    modes = np.arange(1, a.shape[1] + 1, dtype=float)
    x = np.arange(n_grid, dtype=float) / n_grid
    phase = 2.0 * np.pi * modes[:, None] * x[None, :]
    sine_basis = np.sin(phase) / modes[:, None]
    cosine_basis = np.cos(phase) / modes[:, None]
    return amplitude * (a @ sine_basis + b @ cosine_basis)


def solve_burgers(
    initial: np.ndarray, viscosity: float, final_time: float, cfl: float = 0.25
) -> tuple[np.ndarray, int]:
    """Solve u_t+(u^2/2)_x=nu*u_xx using conservative Rusanov flux.

    Spatial discretization: periodic first-order finite volume for advection
    plus centered second-order differences for diffusion.
    Time integration: forward Euler with conservative advective/diffusive steps.
    Both the flux and diffusion terms conserve the spatial mean.
    """
    if initial.ndim != 2:
        raise ValueError("initial must have shape (n_functions, n_grid)")
    if viscosity < 0 or final_time < 0 or not 0 < cfl <= 0.5:
        raise ValueError("invalid PDE parameters or CFL number")
    n_grid = initial.shape[1]
    dx = 1.0 / n_grid
    u = np.asarray(initial, dtype=np.float64).copy()
    t = 0.0
    steps = 0
    while t < final_time:
        max_speed = float(np.max(np.abs(u)))
        adv_dt = cfl * dx / max_speed if max_speed > 1e-12 else np.inf
        diff_dt = cfl * dx * dx / viscosity if viscosity > 0 else np.inf
        dt = min(adv_dt, diff_dt, final_time - t)
        if not np.isfinite(dt) or dt <= 0:
            raise RuntimeError("nonpositive or nonfinite time step")

        right = np.roll(u, -1, axis=1)
        left = np.roll(u, 1, axis=1)
        # Local Lax-Friedrichs / Rusanov flux for f(u)=u^2/2.
        flux_right = (
            0.25 * (u * u + right * right)
            - 0.5 * np.maximum(np.abs(u), np.abs(right)) * (right - u)
        )
        convection = -(flux_right - np.roll(flux_right, 1, axis=1)) / dx
        diffusion = viscosity * (right - 2.0 * u + left) / (dx * dx)
        u += dt * (convection + diffusion)
        t += dt
        steps += 1
        if not np.all(np.isfinite(u)):
            raise FloatingPointError("Burgers solver produced NaN/Inf")
    return u, steps


def relative_l2_per_function(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Return one relative L2 error for each entire solution function."""
    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("prediction and truth must be equal-sized matrices")
    numerator = np.linalg.norm(predicted - truth, axis=1)
    denominator = np.maximum(np.linalg.norm(truth, axis=1), 1e-12)
    return numerator / denominator


def solve_from_coefficients(
    a: np.ndarray, b: np.ndarray, n_grid: int, cfg: SimulationConfig
) -> tuple[np.ndarray, np.ndarray, int]:
    initial = make_initial_conditions(a, b, n_grid, cfg.amplitude)
    solved, steps = solve_burgers(initial, cfg.viscosity, cfg.final_time, cfg.cfl)
    if n_grid % cfg.n_sensors != 0:
        raise ValueError("solver grid must be divisible by n_sensors")
    stride = n_grid // cfg.n_sensors
    return initial[:, ::stride], solved[:, ::stride], steps


def convergence_check(cfg: SimulationConfig) -> dict:
    """Compare identical 24 functions at 128, 256, and 512 grid points."""
    rng = np.random.default_rng(cfg.seed + 1)
    a = rng.standard_normal((24, cfg.n_fourier_modes))
    b = rng.standard_normal((24, cfg.n_fourier_modes))
    solutions = {}
    steps = {}
    for n_grid in (128, 256, 512):
        _, solutions[n_grid], steps[n_grid] = solve_from_coefficients(
            a, b, n_grid, cfg
        )
    e128_256 = relative_l2_per_function(solutions[128], solutions[256])
    e256_512 = relative_l2_per_function(solutions[256], solutions[512])
    results = {
        "sample_size": 24,
        "solver_steps": {str(k): int(v) for k, v in steps.items()},
        "relative_l2_128_vs_256": {
            "median": float(np.median(e128_256)),
            "maximum": float(np.max(e128_256)),
        },
        "relative_l2_256_vs_512": {
            "median": float(np.median(e256_512)),
            "maximum": float(np.max(e256_512)),
        },
        "fraction_with_lower_refinement_error": float(np.mean(e256_512 < e128_256)),
    }
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, 25), e128_256 * 100, "o", ms=4, label="128 vs 256")
    ax.plot(np.arange(1, 25), e256_512 * 100, "s", ms=4, label="256 vs 512")
    ax.set(xlabel="Independent initial condition", ylabel="Relative L2 discrepancy (%)")
    ax.set_title("Grid refinement check on 24 independently drawn functions")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "grid_convergence.png", dpi=180)
    plt.close(fig)
    return results


def generate_dataset(cfg: SimulationConfig, overwrite: bool = False) -> dict:
    if cfg.n_pairs != cfg.n_train + cfg.n_validation + cfg.n_test:
        raise ValueError("split sizes must add to n_pairs")
    if cfg.n_solver % cfg.n_sensors != 0:
        raise ValueError("n_solver must be divisible by n_sensors")
    target_files = [DATA_DIR / f"{split}.npz" for split in
                    ("train", "validation", "test_locked")]
    if any(p.exists() for p in target_files) and not overwrite:
        raise FileExistsError("Dataset exists; use --overwrite to regenerate")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    check = convergence_check(cfg)
    with (RESULTS_DIR / "numerical_validation.json").open("w") as f:
        json.dump(check, f, indent=2)

    rng = np.random.default_rng(cfg.seed)
    a = rng.standard_normal((cfg.n_pairs, cfg.n_fourier_modes))
    b = rng.standard_normal((cfg.n_pairs, cfg.n_fourier_modes))
    initial_grid = make_initial_conditions(a, b, cfg.n_solver, cfg.amplitude)
    final_grid, steps = solve_burgers(
        initial_grid, cfg.viscosity, cfg.final_time, cfg.cfl
    )
    stride = cfg.n_solver // cfg.n_sensors
    X, Y = initial_grid[:, ::stride], final_grid[:, ::stride]

    mass_change = np.max(np.abs(final_grid.mean(axis=1) -
                                initial_grid.mean(axis=1)))
    energy_initial = np.mean(initial_grid ** 2, axis=1)
    energy_final = np.mean(final_grid ** 2, axis=1)
    energy_increases = int(np.sum(energy_final > energy_initial + 1e-12))
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise FloatingPointError("nonfinite dataset values")
    if mass_change > 1e-10 or energy_increases:
        raise AssertionError("conservation/dissipation sanity check failed")

    # Independent coefficient vectors define independent functions.
    indices = rng.permutation(cfg.n_pairs)
    splits = {
        "train": indices[: cfg.n_train],
        "validation": indices[cfg.n_train : cfg.n_train + cfg.n_validation],
        "test_locked": indices[cfg.n_train + cfg.n_validation :],
    }
    for split, idx in splits.items():
        np.savez_compressed(
            DATA_DIR / f"{split}.npz",
            X=X[idx], Y=Y[idx], a=a[idx], b=b[idx],
            x=np.arange(cfg.n_sensors) / cfg.n_sensors,
            sample_id=idx,
        )

    manifest = {
        "pde": "u_t+(u^2/2)_x=nu*u_xx, periodic x in [0,1)",
        "config": asdict(cfg),
        "shapes": {name: [len(idx), cfg.n_sensors] for name, idx in splits.items()},
        "simulator_steps": int(steps),
        "maximum_absolute_mass_change": float(mass_change),
        "number_of_energy_increases": energy_increases,
        "numerical_convergence": check,
        "test_policy": "test_locked.npz must not be loaded during model selection",
    }
    with (DATA_DIR / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def fit_pca_scores(
    X_train: np.ndarray, Y_train: np.ndarray, X_validation: np.ndarray
) -> tuple[PCA, PCA, np.ndarray, np.ndarray, np.ndarray]:
    # IMPORTANT: only training arrays appear in either call to fit_transform.
    input_pca = PCA(n_components=10, svd_solver="full")
    output_pca = PCA(n_components=32, svd_solver="full")
    scores_x_train = input_pca.fit_transform(X_train)
    scores_x_val = input_pca.transform(X_validation)
    scores_y_train = output_pca.fit_transform(Y_train)
    return input_pca, output_pca, scores_x_train, scores_x_val, scores_y_train


def scan_models(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    alphas: tuple[float, ...] = (1e-6, 1e-4, 1e-2, 1.0, 100.0),
) -> tuple[dict, dict, list[dict]]:
    """Tune solely on validation data; never reads any test arrays."""
    _, pca_y, x_train, x_val, y_train = fit_pca_scores(X_train, Y_train, X_val)
    r_inputs = (2, 4, 6, 8, 10)
    r_outputs = (4, 8, 12, 16, 24, 32)
    best = {"PCR": None, "ridge-PCR": None}
    rank_scan = []
    Y_center = pca_y.mean_
    for r_in in r_inputs:
        xtr, xva = x_train[:, :r_in], x_val[:, :r_in]
        candidate_models = [("PCR", 0.0, LinearRegression())]
        candidate_models += [
            ("ridge-PCR", alpha, Ridge(alpha=alpha, solver="svd"))
            for alpha in alphas
        ]
        for name, alpha, model in candidate_models:
            model.fit(xtr, y_train)
            all_output_scores = model.predict(xva)
            for r_out in r_outputs:
                pred = Y_center + (
                    all_output_scores[:, :r_out] @ pca_y.components_[:r_out]
                )
                errors = relative_l2_per_function(pred, Y_val)
                candidate = {
                    "model": name,
                    "r_input": r_in,
                    "r_output": r_out,
                    "alpha": alpha,
                    "validation_error": float(np.mean(errors)),
                    "validation_median_error": float(np.median(errors)),
                }
                rank_scan.append(candidate)
                winner = best[name]
                if winner is None or (
                    candidate["validation_error"], r_in + r_out, alpha
                ) < (
                    winner["validation_error"],
                    winner["r_input"] + winner["r_output"],
                    winner["alpha"],
                ):
                    best[name] = candidate

    # Separate PCA reconstruction floor: this projects observed validation
    # targets and is NOT a deployable predictor or a fitted model score.
    scores_y_val = pca_y.transform(Y_val)
    floor = {}
    for r_out in r_outputs:
        reconstructed = Y_center + (
            scores_y_val[:, :r_out] @ pca_y.components_[:r_out]
        )
        floor[r_out] = float(np.mean(relative_l2_per_function(
            reconstructed, Y_val
        )))
    return best, floor, rank_scan


def refit_selected_and_predict(
    X_train: np.ndarray, Y_train: np.ndarray,
    X_val: np.ndarray, selected: dict
) -> np.ndarray:
    _, pca_y, x_train, x_val, y_train = fit_pca_scores(X_train, Y_train, X_val)
    r_in, r_out = selected["r_input"], selected["r_output"]
    if selected["model"] == "PCR":
        reg = LinearRegression()
    else:
        reg = Ridge(alpha=selected["alpha"], solver="svd")
    reg.fit(x_train[:, :r_in], y_train)
    latent = reg.predict(x_val[:, :r_in])
    return pca_y.mean_ + latent[:, :r_out] @ pca_y.components_[:r_out]


def plot_learning_curve(details: pd.DataFrame, X_val: np.ndarray,
                        Y_val: np.ndarray) -> None:
    summary = (
        details.groupby(["n_train", "model"], as_index=False)
        .agg(mean=("validation_error", "mean"),
             sd=("validation_error", "std"))
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, color, shift, marker in (
        ("PCR", "tab:blue", 0.975, "o"),
        ("ridge-PCR", "tab:orange", 1.025, "s"),
    ):
        subset = summary[summary["model"] == name].sort_values("n_train")
        xs = subset["n_train"].to_numpy()
        ys = subset["mean"].to_numpy()
        std = subset["sd"].to_numpy()
        # Tiny horizontal offsets reveal both methods when errors overlap.
        ax.errorbar(xs * shift, ys, yerr=std, color=color, marker=marker,
                    capsize=3, linewidth=1.5, ls="--" if name == "ridge-PCR"
                    else "-", label=f"{name} (mean +/- 1 seed SD)")
    persistence = float(np.mean(relative_l2_per_function(X_val, Y_val)))
    ax.text(0.03, 0.97, f"No-evolution reference: {persistence:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.9})
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(details["n_train"].unique()))
    ax.set_xticklabels([str(n) for n in sorted(details["n_train"].unique())])
    ax.set(xlabel="Number of training initial conditions",
           ylabel="Mean relative L2 error on validation functions",
           title="Burgers operator: validation learning curves")
    ax.set_ylim(0.295, 0.41)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_learning_curves.png", dpi=180)
    plt.close(fig)


def plot_rank_sensitivity(rank_scan: list[dict]) -> None:
    df = pd.DataFrame(rank_scan)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, name in zip(axes, ("PCR", "ridge-PCR")):
        subset = df[df["model"] == name]
        matrix = subset.pivot_table(
            index="r_input", columns="r_output",
            values="validation_error", aggfunc="min"
        ).sort_index()
        im = ax.imshow(matrix.to_numpy(), origin="lower", aspect="auto",
                       cmap="viridis")
        ax.set_xticks(range(matrix.shape[1]), matrix.columns.astype(str))
        ax.set_yticks(range(matrix.shape[0]), matrix.index.astype(str))
        ax.set(xlabel="Output PCA components", ylabel="Input PCA components",
               title=f"{name} validation error")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rank_sensitivity.png", dpi=180)
    plt.close(fig)


def plot_example_predictions(
    X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray,
    Y_val: np.ndarray, selected: dict
) -> None:
    xs = np.arange(X_val.shape[1]) / X_val.shape[1]
    pred_pcr = refit_selected_and_predict(X_train, Y_train, X_val, selected["PCR"])
    pred_ridge = refit_selected_and_predict(
        X_train, Y_train, X_val, selected["ridge-PCR"]
    )
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(xs, X_val[i], color="gray", ls=":", label="Initial condition")
        ax.plot(xs, Y_val[i], color="black", label="Solver truth")
        ax.plot(xs, pred_pcr[i], color="tab:blue", ls="--", label="PCR")
        ax.plot(xs, pred_ridge[i], color="tab:orange", ls="-.", label="ridge-PCR")
        ax.set(ylabel="u(x)", title=f"Fixed validation example {i + 1}")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Spatial position x")
    axes[0].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "validation_examples.png", dpi=180)
    plt.close(fig)


def run_baselines() -> pd.DataFrame:
    train_path = DATA_DIR / "train.npz"
    val_path = DATA_DIR / "validation.npz"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Run 'generate' before 'baselines'")
    # Deliberately never load data/test_locked.npz.
    with np.load(train_path) as train, np.load(val_path) as val:
        X_train, Y_train = train["X"].copy(), train["Y"].copy()
        X_val, Y_val = val["X"].copy(), val["Y"].copy()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    details, all_scans = [], None
    sizes = (50, 100, 200, 400, 800)
    seeds = (4020, 4021, 4022, 4023, 4024)
    selected_for_examples = None
    start = time.perf_counter()
    for seed in seeds:
        # The same nested function subsets are used for both regressors.
        ordering = np.random.default_rng(seed).permutation(len(X_train))
        for n in sizes:
            subset = ordering[:n]
            best, floor, scan = scan_models(
                X_train[subset], Y_train[subset], X_val, Y_val
            )
            for name in ("PCR", "ridge-PCR"):
                details.append({
                    "seed": seed, "n_train": n, **best[name],
                    "output_pca_floor_at_chosen_rank": floor[best[name]["r_output"]],
                })
            print(
                f"seed {seed} n={n:3d}: PCR={best['PCR']['validation_error']:.5f}, "
                f"ridge-PCR={best['ridge-PCR']['validation_error']:.5f}",
                flush=True,
            )
            if seed == 4020 and n == 800:
                all_scans = scan
                selected_for_examples = (subset.copy(), best)

    df = pd.DataFrame(details)
    df.to_csv(RESULTS_DIR / "best_validation_models.csv", index=False)
    summary = (
        df.groupby(["n_train", "model"], as_index=False)
        .agg(mean_relative_l2=("validation_error", "mean"),
             sd_across_seeds=("validation_error", "std"),
             min_across_seeds=("validation_error", "min"),
             max_across_seeds=("validation_error", "max"))
    )
    summary.to_csv(RESULTS_DIR / "validation_summary.csv", index=False)
    plot_learning_curve(df, X_val, Y_val)
    if all_scans is not None:
        pd.DataFrame(all_scans).to_csv(
            RESULTS_DIR / "full_rank_scan_n800_seed4020.csv", index=False
        )
        plot_rank_sensitivity(all_scans)
    if selected_for_examples is not None:
        subset, chosen = selected_for_examples
        plot_example_predictions(
            X_train[subset], Y_train[subset], X_val, Y_val, chosen
        )

    seconds = time.perf_counter() - start
    persistence = float(np.mean(relative_l2_per_function(X_val, Y_val)))
    rows_800 = summary[summary["n_train"] == 800]
    pcr = float(rows_800.loc[
        rows_800["model"] == "PCR", "mean_relative_l2"
    ].iloc[0])
    ridge = float(rows_800.loc[
        rows_800["model"] == "ridge-PCR", "mean_relative_l2"
    ].iloc[0])
    improvement = 100.0 * (pcr - ridge) / pcr
    vs_persistence = 100.0 * (persistence - pcr) / persistence
    pcr_50 = float(summary.loc[
        (summary["n_train"] == 50) & (summary["model"] == "PCR"),
        "mean_relative_l2",
    ].iloc[0])
    vs_50 = 100.0 * (pcr_50 - pcr) / pcr_50
    rank_row = df[(df["seed"] == 4020) & (df["n_train"] == 800) &
                  (df["model"] == "PCR")].iloc[0]
    result_lines = []
    for n in sizes:
        p = summary[(summary["n_train"] == n) & (summary["model"] == "PCR")].iloc[0]
        r = summary[
            (summary["n_train"] == n) & (summary["model"] == "ridge-PCR")
        ].iloc[0]
        result_lines.append(
            f"| {n} | {p['mean_relative_l2']:.4f} +/- "
            f"{p['sd_across_seeds']:.4f} | {r['mean_relative_l2']:.4f} +/- "
            f"{r['sd_across_seeds']:.4f} |"
        )
    result_table = "\n".join(result_lines)
    report = f"""# Burgers solution operator: baseline checkpoint

## Scope and data
- Governing PDE: viscous Burgers equation, periodic [0,1), viscosity 0.01, T=0.25.
- Simulated pairs: 1200 (800 train, 200 validation, 200 locked test), 64 sensors each.
- Training subset sizes: {list(sizes)}; repeated seeds: {list(seeds)}.
- **Test policy:** the locked test file was never read during baseline model selection.

## Validation findings (not final test results)
- Persistence (predict u(x,T)=u(x,0)) relative L2 error: {persistence:.5f}.
- At N=800, PCR mean validation relative L2 across five subsets: {pcr:.5f}.
- At N=800, ridge-PCR mean validation relative L2: {ridge:.5f}.
- Ridge improvement against PCR at N=800: {improvement:+.2f}% (positive favors ridge).
- PCR improves by {vs_persistence:.1f}% over persistence at N=800.
- PCR error decreases by {vs_50:.1f}% between N=50 and N=800.
- For the N=800, seed-4020 PCR fit: input rank {int(rank_row['r_input'])},
  output rank {int(rank_row['r_output'])}, and output-PCA projection floor
  {rank_row['output_pca_floor_at_chosen_rank']:.4f} on validation.
- Baseline analysis runtime: {seconds:.1f} seconds.

| Training functions N | PCR mean +/- seed SD | ridge-PCR mean +/- seed SD |
| ---: | ---: | ---: |
{result_table}

See `validation_summary.csv` for all N, means, and seed-to-seed standard
deviations. The error bars in `figures/validation_learning_curves.png`
show **one standard deviation across five training subsets**, not 95% CIs.

## Interpretation and cautions
- PCR and ridge-PCR are both linear mappings between learned PCA coefficients.
- Input functions lie in a 10-dimensional Fourier subspace; therefore the
  input PCA search is restricted to 2, 4, 6, 8, or 10 components.
- Nonlinear Burgers evolution can generate output modes beyond those 10 inputs;
  output rank is separately tuned among 4, 8, 12, 16, 24, and 32 components.
- Nearly identical ridge/PCR errors suggest the dominant limitation is the
  **linear input-to-output mapping** rather than overfitting in this setting.
  The PCA projection floor is below total PCR prediction error, consistent
  with a gap that improved linear fitting alone may not close.
- Validation error decreases noticeably up to 200 functions but levels off
  between 400 and 800. DeepONet will test whether a nonlinear operator
  approximation can reduce this apparent error floor.
- At N=800, all five subset seeds contain the same complete training pool.
  Their near-zero SD is a design consequence, **not** evidence of zero
  uncertainty. At smaller N the SD measures sensitivity to subsampling
  this fixed pool, not variation across independently simulated datasets.
- Model selection and reported performance both use the validation set; scores
  are exploratory and can be optimistic. Only the planned final evaluation
  on untouched test functions can support generalization claims.
- No DeepONet was trained in this checkpoint; that is the next comparison.

## Next steps
1. Share the data format and training/validation splits with Sudarshan.
2. Train DeepONet on the same five nested subset sizes and evaluate on validation.
3. Freeze architectures, PCA ranks, penalties, and other tuning decisions.
4. Evaluate all frozen models on the locked test functions once, near Nov 30.
5. Compute paired function-level bootstrap intervals on *test* differences.
"""
    (RESULTS_DIR / "baseline_checkpoint.md").write_text(report)
    print(f"Baseline analysis completed in {seconds:.1f} s.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("generate", "baselines", "all"),
        help="generate data, fit baselines, or do both",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow regeneration of the already saved dataset",
    )
    args = parser.parse_args()
    if args.command in ("generate", "all"):
        cfg = SimulationConfig()
        manifest = generate_dataset(cfg, overwrite=args.overwrite)
        print(json.dumps({
            "shapes": manifest["shapes"],
            "mass_change": manifest["maximum_absolute_mass_change"],
            "convergence": manifest["numerical_convergence"],
        }, indent=2))
    if args.command in ("baselines", "all"):
        run_baselines()


if __name__ == "__main__":
    main()
