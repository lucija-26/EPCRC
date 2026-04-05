"""Shared UTD19 data loading, preprocessing, and per-city model training.

Both `experiment_0forward` and `experiment_0backward` call into this module.
Trained models and response matrices (Y_fit / Y_eval) are cached on disk so
that the second experiment reuses the first experiment's work.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Default configuration for UTD19 experiments
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # Data subsetting (keeps training feasible on a laptop)
    horizon_steps=12,
    lag_steps=[1, 2, 3, 6],
    max_days_per_city=14,
    max_detectors_per_city=40,
    min_points_per_detector=2000,
    # Train/val/test filtering
    min_train=500,
    min_val=200,
    min_test=200,
    # Response-matrix sampling
    max_fit_samples=1500,
    max_eval_samples=1500,
    # sklearn HGB model
    model_params=dict(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=0.0,
    ),
    seed=0,
)


def set_global_seed(seed: int) -> np.random.RandomState:
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)
    return rng


# ---------------------------------------------------------------------------
# Data preparation (schema: day, interval, detid, flow, occ, error, city, speed)
# ---------------------------------------------------------------------------
def prepare_utd19(
    df_raw: pd.DataFrame,
    horizon_steps: int,
    lag_steps: List[int],
    max_days_per_city: Optional[int],
    max_detectors_per_city: Optional[int],
    min_points_per_detector: int,
) -> Tuple[pd.DataFrame, List[str]]:
    df = df_raw.copy()
    df["day"] = pd.to_datetime(df["day"], errors="coerce")
    df = df.dropna(subset=["day"])

    for c in ["interval", "flow", "occ"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["interval", "flow", "occ"])

    if "error" in df.columns:
        df["error"] = pd.to_numeric(df["error"], errors="coerce").fillna(0.0)
        df = df[df["error"] == 0.0].copy()

    # Keep only most-recent max_days_per_city per city
    if max_days_per_city is not None:
        max_day = df.groupby("city")["day"].transform("max")
        cutoff = max_day - pd.to_timedelta(max_days_per_city - 1, unit="D")
        df = df[df["day"] >= cutoff].copy()

    # Filter detectors by record count, optionally cap per city
    counts = df.groupby(["city", "detid"], sort=False).size().reset_index(name="_n")
    counts = counts[counts["_n"] >= min_points_per_detector]
    if max_detectors_per_city is not None:
        keep = []
        for _, sub in counts.groupby("city", sort=False):
            keep.append(sub.nlargest(max_detectors_per_city, "_n")[["city", "detid"]])
        counts = pd.concat(keep, ignore_index=True)
    else:
        counts = counts[["city", "detid"]]
    df = df.merge(counts, on=["city", "detid"], how="inner")

    df = df.sort_values(["city", "detid", "day", "interval"])

    # Calendar features
    df["dow"] = df["day"].dt.dayofweek.astype(np.int8)
    df["is_weekend"] = (df["dow"] >= 5).astype(np.int8)
    df["dow_sin"] = np.sin(2.0 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2.0 * np.pi * df["dow"] / 7.0)

    interval_max = df.groupby("city")["interval"].transform("max")
    interval_period = (interval_max + 1.0).replace(0.0, 1.0)
    tod_frac = (df["interval"] / interval_period).clip(0.0, 1.0)
    df["tod_sin"] = np.sin(2.0 * np.pi * tod_frac)
    df["tod_cos"] = np.cos(2.0 * np.pi * tod_frac)

    g = df.groupby(["city", "detid"], sort=False)
    lag_steps_unique = sorted({int(l) for l in lag_steps})
    for lag in lag_steps_unique:
        df[f"flow_lag{lag}"] = g["flow"].shift(lag)
        df[f"occ_lag{lag}"] = g["occ"].shift(lag)

    if 1 not in lag_steps_unique:
        df["flow_lag1"] = g["flow"].shift(1)
        lag_steps_unique = sorted(set(lag_steps_unique + [1]))
    df["flow_diff1"] = df["flow"] - df["flow_lag1"]

    df["y"] = g["flow"].shift(-horizon_steps)

    df["_t_idx"] = g.cumcount()
    df["_t_len"] = g["flow"].transform("size")
    train_end = (0.6 * df["_t_len"]).astype(int)
    val_end = (0.8 * df["_t_len"]).astype(int)
    df["split"] = None
    df.loc[df["_t_idx"] < (train_end - horizon_steps), "split"] = "train"
    df.loc[(df["_t_idx"] >= train_end) & (df["_t_idx"] < (val_end - horizon_steps)), "split"] = "val"
    df.loc[(df["_t_idx"] >= val_end) & (df["_t_idx"] < (df["_t_len"] - horizon_steps)), "split"] = "test"

    feature_cols = (
        ["interval", "flow", "occ"]
        + ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "flow_diff1"]
        + [f"flow_lag{l}" for l in lag_steps_unique]
        + [f"occ_lag{l}" for l in lag_steps_unique]
    )
    df = df.dropna(subset=feature_cols + ["y", "split"]).copy()
    df = df.drop(columns=["_t_idx", "_t_len"], errors="ignore")
    return df, feature_cols


def build_xy(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=float, copy=True)
    y = df["y"].to_numpy(dtype=float, copy=True)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X[mask], y[mask]


# ---------------------------------------------------------------------------
# Build response matrices, with on-disk caching
# ---------------------------------------------------------------------------
@dataclass
class Utd19Bundle:
    model_names: List[str]
    Y_fit: np.ndarray           # (n_fit, n_models)
    Y_eval: np.ndarray          # (n_eval, n_models)
    per_city_rmse: Dict[str, float]  # train-set diagnostic, for reporting


def _load_raw_csv(data_csv: str, chunked: bool = True) -> pd.DataFrame:
    if not chunked:
        return pd.read_csv(data_csv, low_memory=False)
    use_cols = ["day", "interval", "detid", "flow", "occ", "error", "city"]
    chunks = []
    for ch in pd.read_csv(data_csv, usecols=use_cols, chunksize=2_000_000, low_memory=False):
        chunks.append(ch)
    return pd.concat(chunks, ignore_index=True)


def build_or_load_bundle(
    data_csv: str,
    cache_dir: str,
    config: Optional[dict] = None,
    cities: Optional[List[str]] = None,
    verbose: bool = True,
) -> Utd19Bundle:
    """Load data, train per-city models, build Y_fit/Y_eval. Cache everything."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    os.makedirs(cache_dir, exist_ok=True)
    models_dir = os.path.join(cache_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    bundle_path = os.path.join(cache_dir, "bundle.npz")

    if os.path.exists(bundle_path):
        if verbose:
            print(f"[cache] loading Y_fit / Y_eval from {bundle_path}")
        with np.load(bundle_path, allow_pickle=True) as data:
            return Utd19Bundle(
                model_names=list(data["model_names"]),
                Y_fit=data["Y_fit"],
                Y_eval=data["Y_eval"],
                per_city_rmse=dict(data["per_city_rmse"].item()),
            )

    rng = set_global_seed(cfg["seed"])

    if verbose:
        print(f"[data] loading {data_csv} ...")
    df_raw = _load_raw_csv(data_csv, chunked=True)
    if verbose:
        print(f"[data] raw rows: {len(df_raw):,}")

    if cities is not None:
        df_raw = df_raw[df_raw["city"].isin(cities)].copy()
        if verbose:
            print(f"[data] restricted to {len(cities)} cities -> {len(df_raw):,} rows")

    df, feature_cols = prepare_utd19(
        df_raw,
        horizon_steps=cfg["horizon_steps"],
        lag_steps=cfg["lag_steps"],
        max_days_per_city=cfg["max_days_per_city"],
        max_detectors_per_city=cfg["max_detectors_per_city"],
        min_points_per_detector=cfg["min_points_per_detector"],
    )
    if verbose:
        print(f"[data] after preprocessing: {len(df):,} rows, {len(feature_cols)} features")

    del df_raw

    # Train / load one model per city
    city_models: Dict[str, HistGradientBoostingRegressor] = {}
    city_data: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    city_train_rmse: Dict[str, float] = {}

    cities_present = sorted(df["city"].dropna().unique().tolist())
    if verbose:
        print(f"[train] training/loading {len(cities_present)} city models")

    for city in cities_present:
        dc = df[df["city"] == city]
        X_train, y_train = build_xy(dc[dc["split"] == "train"], feature_cols)
        X_val, y_val = build_xy(dc[dc["split"] == "val"], feature_cols)
        X_test, y_test = build_xy(dc[dc["split"] == "test"], feature_cols)

        if (
            X_train.shape[0] < cfg["min_train"]
            or X_val.shape[0] < cfg["min_val"]
            or X_test.shape[0] < cfg["min_test"]
        ):
            if verbose:
                print(
                    f"  {city}: SKIP (train={X_train.shape[0]}, "
                    f"val={X_val.shape[0]}, test={X_test.shape[0]})"
                )
            continue

        cache_path = os.path.join(models_dir, f"{city.replace('/', '_')}.joblib")
        if os.path.exists(cache_path):
            model = joblib.load(cache_path)
            src = "cached"
        else:
            model = HistGradientBoostingRegressor(
                **cfg["model_params"], random_state=cfg["seed"]
            )
            model.fit(X_train, y_train)
            joblib.dump(model, cache_path)
            src = "trained"

        city_models[city] = model
        city_data[city] = {"val": (X_val, y_val), "test": (X_test, y_test)}
        train_pred = model.predict(X_train)
        rmse = float(np.sqrt(np.mean((train_pred - y_train) ** 2)))
        city_train_rmse[city] = rmse
        if verbose:
            print(f"  {city}: {src}  train_rmse={rmse:.2f}  n_train={X_train.shape[0]}")

    del df

    model_names = sorted(city_models.keys())
    N = len(model_names)
    if verbose:
        print(f"[train] {N} city models ready")
    if N < 3:
        raise RuntimeError(f"Need at least 3 city models, got {N}")

    # Pool & subsample shared query points
    X_val_all = np.concatenate([city_data[c]["val"][0] for c in model_names], axis=0)
    X_test_all = np.concatenate([city_data[c]["test"][0] for c in model_names], axis=0)

    fit_n = min(cfg["max_fit_samples"], X_val_all.shape[0])
    eval_n = min(cfg["max_eval_samples"], X_test_all.shape[0])

    fit_idx = rng.choice(X_val_all.shape[0], size=fit_n, replace=False)
    eval_idx = rng.choice(X_test_all.shape[0], size=eval_n, replace=False)

    X_fit = X_val_all[fit_idx]
    X_eval = X_test_all[eval_idx]

    if verbose:
        print(f"[resp] building response matrices: fit={fit_n}, eval={eval_n}")
    Y_fit = np.column_stack([city_models[c].predict(X_fit) for c in model_names])
    Y_eval = np.column_stack([city_models[c].predict(X_eval) for c in model_names])

    np.savez_compressed(
        bundle_path,
        model_names=np.array(model_names, dtype=object),
        Y_fit=Y_fit,
        Y_eval=Y_eval,
        per_city_rmse=np.array(city_train_rmse, dtype=object),
    )
    if verbose:
        print(f"[cache] saved bundle to {bundle_path}")

    return Utd19Bundle(
        model_names=model_names,
        Y_fit=Y_fit,
        Y_eval=Y_eval,
        per_city_rmse=city_train_rmse,
    )
