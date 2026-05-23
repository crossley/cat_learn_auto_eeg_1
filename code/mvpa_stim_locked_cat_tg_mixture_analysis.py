#!/usr/bin/env python3
"""Bayesian bivariate-Gaussian mixture analysis of stim-locked TG matrices.

Three models are fit per (train_day, test_day) pair to the stack of
subject-level AUC matrices loaded from the precomputed NPZ files:

  Model 1  - constant baseline (no spatial structure)
  Model 2  - baseline + one bivariate-normal component
  Model 3  - baseline + two bivariate-normal components; label switching is
             broken by a soft potential keeping component 1 closer to the
             main diagonal than component 2

Outputs written to OUTPUT_DIR:
  mvpa_stim_locked_cat_tg_mixture_component_summary.csv
  mvpa_stim_locked_cat_tg_mixture_fitted_surface.csv
  mvpa_stim_locked_cat_tg_mixture_trace_trainD{n}_testD{m}_{model}.nc
  mvpa_stim_locked_cat_tg_mixture_progress.json
"""

import json
import logging
import os
import re
import time
import warnings
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

logging.getLogger("pymc").setLevel(logging.ERROR)
logging.getLogger("pytensor").setLevel(logging.ERROR)
logging.getLogger("arviz").setLevel(logging.ERROR)

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.special import expit

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

SUBSAMPLE_STEP = 4
N_CHAINS = 2
N_TUNE = 1000
N_DRAWS = 1000
RANDOM_SEED = 42
SAVE_TRACES = True
DAY_GRID = [1, 2, 3, 4, 5]

_NPZ_RE = re.compile(
    r"mvpa_stim_locked_cat_tg_matrix_sub_(\d+)_trainD(\d+)_testD(\d+)\.npz$"
)


def load_day_pair_matrices(output_dir, train_day, test_day, subsample_step):
    paths = sorted(
        output_dir.glob(
            f"mvpa_stim_locked_cat_tg_matrix_sub_*"
            f"_trainD{train_day}_testD{test_day}.npz"
        )
    )
    if not paths:
        raise FileNotFoundError(
            f"No NPZ files for trainD{train_day}_testD{test_day} in {output_dir}"
        )
    mats, subjects, t_sub = [], [], None
    for path in paths:
        m = _NPZ_RE.search(path.name)
        if m is None:
            continue
        with np.load(path, allow_pickle=False) as z:
            auc = z["auc"][::subsample_step, ::subsample_step].astype(float)
            t = z["time_sec"][::subsample_step].astype(float)
        if t_sub is None:
            t_sub = t
        mats.append(auc)
        subjects.append(int(m.group(1)))
    X = np.clip(np.stack(mats, axis=0), 1e-5, 1.0 - 1e-5)
    return X, t_sub, subjects


def _bvn_surface_pt(train_t, test_t, mu_x, mu_y, log_sx, log_sy, rho_raw):
    sx = pt.exp(log_sx)
    sy = pt.exp(log_sy)
    rho = pt.tanh(rho_raw)
    dx = (train_t[:, None] - mu_x) / sx
    dy = (test_t[None, :] - mu_y) / sy
    z = (dx ** 2 - 2.0 * rho * dx * dy + dy ** 2) / (2.0 * (1.0 - rho ** 2 + 1e-6))
    return pt.exp(-z)


def _bvn_surface_np(t_sub, mu_x, mu_y, log_sx, log_sy, rho_raw):
    """Vectorised numpy BVN surface.

    Parameters
    ----------
    t_sub : (n,) array
    mu_x, mu_y, log_sx, log_sy, rho_raw : (n_samples,) arrays

    Returns
    -------
    (n_samples, n, n) array
    """
    sx = np.exp(log_sx)
    sy = np.exp(log_sy)
    rho = np.tanh(rho_raw)
    dx = (t_sub[None, :] - mu_x[:, None]) / sx[:, None]
    dy = (t_sub[None, :] - mu_y[:, None]) / sy[:, None]
    z = (
        dx[:, :, None] ** 2
        - 2.0 * rho[:, None, None] * dx[:, :, None] * dy[:, None, :]
        + dy[:, None, :] ** 2
    ) / (2.0 * (1.0 - rho[:, None, None] ** 2 + 1e-6))
    return np.exp(-z)


def build_model_1(X, t_sub):
    with pm.Model() as model:
        baseline = pm.Normal("baseline", mu=0.0, sigma=1.0)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.02)
        pm.Beta("obs", mu=pm.math.sigmoid(baseline), nu=nu, observed=X)
    return model


def build_model_2(X, t_sub):
    t_min, t_max = float(t_sub.min()), float(t_sub.max())
    train_t = pt.as_tensor_variable(t_sub)
    test_t = pt.as_tensor_variable(t_sub)
    with pm.Model() as model:
        baseline = pm.Normal("baseline", mu=0.0, sigma=1.0)
        amplitude = pm.HalfNormal("amplitude", sigma=1.0)
        mu_x = pm.Uniform("mu_x", lower=t_min, upper=t_max)
        mu_y = pm.Uniform("mu_y", lower=t_min, upper=t_max)
        log_sx = pm.Normal("log_sx", mu=float(np.log(0.1)), sigma=0.5)
        log_sy = pm.Normal("log_sy", mu=float(np.log(0.1)), sigma=0.5)
        rho_raw = pm.Normal("rho_raw", mu=0.0, sigma=0.5)
        surface = _bvn_surface_pt(train_t, test_t, mu_x, mu_y, log_sx, log_sy, rho_raw)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.02)
        pm.Beta(
            "obs",
            mu=pm.math.sigmoid(baseline + amplitude * surface),
            nu=nu,
            observed=X,
        )
    return model


def build_model_3(X, t_sub):
    t_min, t_max = float(t_sub.min()), float(t_sub.max())
    train_t = pt.as_tensor_variable(t_sub)
    test_t = pt.as_tensor_variable(t_sub)
    with pm.Model() as model:
        baseline = pm.Normal("baseline", mu=0.0, sigma=1.0)
        amplitude_1 = pm.HalfNormal("amplitude_1", sigma=1.0)
        mu_x_1 = pm.Uniform("mu_x_1", lower=t_min, upper=t_max)
        mu_y_1 = pm.Uniform("mu_y_1", lower=t_min, upper=t_max)
        log_sx_1 = pm.Normal("log_sx_1", mu=float(np.log(0.1)), sigma=0.5)
        log_sy_1 = pm.Normal("log_sy_1", mu=float(np.log(0.1)), sigma=0.5)
        rho_raw_1 = pm.Normal("rho_raw_1", mu=0.0, sigma=0.5)
        amplitude_2 = pm.HalfNormal("amplitude_2", sigma=1.0)
        mu_x_2 = pm.Uniform("mu_x_2", lower=t_min, upper=t_max)
        mu_y_2 = pm.Uniform("mu_y_2", lower=t_min, upper=t_max)
        log_sx_2 = pm.Normal("log_sx_2", mu=float(np.log(0.1)), sigma=0.5)
        log_sy_2 = pm.Normal("log_sy_2", mu=float(np.log(0.1)), sigma=0.5)
        rho_raw_2 = pm.Normal("rho_raw_2", mu=0.0, sigma=0.5)
        pm.Potential(
            "label_order",
            pt.switch(
                pt.lt(pt.abs(mu_x_1 - mu_y_1), pt.abs(mu_x_2 - mu_y_2)),
                0.0,
                -100.0,
            ),
        )
        s1 = _bvn_surface_pt(train_t, test_t, mu_x_1, mu_y_1, log_sx_1, log_sy_1, rho_raw_1)
        s2 = _bvn_surface_pt(train_t, test_t, mu_x_2, mu_y_2, log_sx_2, log_sy_2, rho_raw_2)
        nu = pm.Gamma("nu", alpha=2.0, beta=0.02)
        pm.Beta(
            "obs",
            mu=pm.math.sigmoid(baseline + amplitude_1 * s1 + amplitude_2 * s2),
            nu=nu,
            observed=X,
        )
    return model


def compute_posterior_mean_surface(idata, model_name, t_sub):
    post = idata.posterior

    def flat(name):
        return post[name].values.reshape(-1)

    baseline = flat("baseline")
    n = len(t_sub)

    if model_name == "model_1":
        return np.full((n, n), float(expit(baseline).mean()))

    if model_name == "model_2":
        s = _bvn_surface_np(
            t_sub, flat("mu_x"), flat("mu_y"),
            flat("log_sx"), flat("log_sy"), flat("rho_raw"),
        )
        eta = baseline[:, None, None] + flat("amplitude")[:, None, None] * s
        return expit(eta).mean(axis=0)

    if model_name == "model_3":
        s1 = _bvn_surface_np(
            t_sub, flat("mu_x_1"), flat("mu_y_1"),
            flat("log_sx_1"), flat("log_sy_1"), flat("rho_raw_1"),
        )
        s2 = _bvn_surface_np(
            t_sub, flat("mu_x_2"), flat("mu_y_2"),
            flat("log_sx_2"), flat("log_sy_2"), flat("rho_raw_2"),
        )
        eta = (
            baseline[:, None, None]
            + flat("amplitude_1")[:, None, None] * s1
            + flat("amplitude_2")[:, None, None] * s2
        )
        return expit(eta).mean(axis=0)

    raise ValueError(f"Unknown model_name: {model_name}")


def _summary_rows(idata, model_name, train_day, test_day):
    exclude = {"label_order"}
    var_names = []
    for var_name in idata.posterior.data_vars:
        if var_name not in exclude:
            var_names.append(var_name)
    df = az.summary(idata, var_names=var_names, hdi_prob=0.94)
    component_map = {
        "baseline": 0, "nu": 0,
        "amplitude": 1, "mu_x": 1, "mu_y": 1,
        "log_sx": 1, "log_sy": 1, "rho_raw": 1,
        "amplitude_1": 1, "mu_x_1": 1, "mu_y_1": 1,
        "log_sx_1": 1, "log_sy_1": 1, "rho_raw_1": 1,
        "amplitude_2": 2, "mu_x_2": 2, "mu_y_2": 2,
        "log_sx_2": 2, "log_sy_2": 2, "rho_raw_2": 2,
    }
    rows = []
    for param, row in df.iterrows():
        rows.append({
            "train_day": train_day,
            "test_day": test_day,
            "model": model_name,
            "component": component_map.get(str(param), 0),
            "param": str(param),
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "hdi_3pct": float(row.get("hdi_3%", np.nan)),
            "hdi_97pct": float(row.get("hdi_97%", np.nan)),
            "r_hat": float(row.get("r_hat", np.nan)),
        })
    return rows


def _surface_rows(surface_mean, model_name, train_day, test_day, t_sub):
    rows = []
    for ti, tt in enumerate(t_sub):
        for tj, tv in enumerate(t_sub):
            rows.append({
                "train_day": train_day,
                "test_day": test_day,
                "model": model_name,
                "train_time_sec": float(tt),
                "test_time_sec": float(tv),
                "mu_mean": float(surface_mean[ti, tj]),
            })
    return rows


def run_mvpa_stim_locked_cat_tg_mixture(
    output_dir=OUTPUT_DIR,
    subsample_step=SUBSAMPLE_STEP,
    n_chains=N_CHAINS,
    n_tune=N_TUNE,
    n_draws=N_DRAWS,
    random_seed=RANDOM_SEED,
    save_traces=SAVE_TRACES,
    day_grid=DAY_GRID,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = output_dir / "mvpa_stim_locked_cat_tg_mixture_component_summary.csv"
    surface_csv = output_dir / "mvpa_stim_locked_cat_tg_mixture_fitted_surface.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_tg_mixture_progress.json"

    wrote_summary = False
    wrote_surface = False
    t0 = time.time()

    day_pairs = []
    for td in day_grid:
        for vd in day_grid:
            day_pairs.append((td, vd))
    model_specs = [
        ("model_1", build_model_1),
        ("model_2", build_model_2),
        ("model_3", build_model_3),
    ]

    n_done = 0
    for train_day, test_day in day_pairs:
        X, t_sub, subjects = load_day_pair_matrices(
            output_dir, train_day, test_day, subsample_step
        )

        for model_name, builder in model_specs:
            trace_path = (
                output_dir
                / f"mvpa_stim_locked_cat_tg_mixture_trace"
                  f"_trainD{train_day}_testD{test_day}_{model_name}.nc"
            )
            if save_traces and trace_path.exists():
                idata = az.from_netcdf(str(trace_path))
            else:
                model = builder(X, t_sub)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with model:
                        idata = pm.sample(
                            draws=n_draws,
                            tune=n_tune,
                            chains=n_chains,
                            cores=n_chains,
                            random_seed=random_seed,
                            progressbar=False,
                        )
                if save_traces:
                    idata.to_netcdf(str(trace_path))

            summary_chunk = _summary_rows(idata, model_name, train_day, test_day)
            pd.DataFrame(summary_chunk).to_csv(
                summary_csv, mode="a", header=not wrote_summary, index=False
            )
            wrote_summary = True

            surface_mean = compute_posterior_mean_surface(idata, model_name, t_sub)
            surface_chunk = _surface_rows(surface_mean, model_name, train_day, test_day, t_sub)
            pd.DataFrame(surface_chunk).to_csv(
                surface_csv, mode="a", header=not wrote_surface, index=False
            )
            wrote_surface = True

        n_done += 1
        elapsed = time.time() - t0
        progress = {
            "n_done": n_done,
            "n_total": len(day_pairs),
            "elapsed_min": round(elapsed / 60.0, 2),
        }
        progress_json.write_text(json.dumps(progress, indent=2))
        print(
            f"[TG mixture] {n_done}/{len(day_pairs)} day-pairs done "
            f"(trainD{train_day} testD{test_day}, {elapsed/60:.1f} min)",
            flush=True,
        )

    return {
        "summary_csv": summary_csv,
        "surface_csv": surface_csv,
        "progress_json": progress_json,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_mixture()
