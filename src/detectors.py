"""Four unsupervised anomaly detectors plus an ensemble vote.

1. STL decomposition + rolling robust z-score   (statistical baseline)
2. Forecast residual: gradient boosting on calendar features (ML, cross-fitted)
3. Isolation Forest on load + change features    (ML, multivariate)
4. Autoencoder on 24-hour daily profiles         (neural network, day-level shape)
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

WINDOW = 24 * 30  # 30-day rolling window for robust scaling


def robust_z(x: pd.Series, window: int = WINDOW) -> pd.Series:
    """Rolling robust z-score using median and MAD (resistant to the anomalies themselves)."""
    mp = window // 4
    med = x.rolling(window, center=True, min_periods=mp).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=mp).median()
    return (x - med) / (1.4826 * mad + 1e-9)


# ---------------------------------------------------------------- 1. STL
def stl_detector(df: pd.DataFrame, z_thresh: float = 4.0) -> pd.DataFrame:
    # Weekly season (168h) contains the daily pattern. Non-robust STL is ~10x faster on
    # 145k points; robustness comes from the median/MAD scoring of the residual instead.
    res = STL(df["load_mw"], period=168, robust=False).fit()
    resid = pd.Series(res.resid, index=df.index)
    z = robust_z(resid)
    return pd.DataFrame({
        "stl_expected": df["load_mw"] - resid,
        "stl_z": z,
        "stl_flag": z.abs() > z_thresh,
    })


# ---------------------------------------------------- 2. Forecast residual
FORECAST_FEATURES = ["hour", "dayofweek", "month", "dayofyear", "year", "is_holiday",
                     "is_weekend", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]
WEATHER_FEATURES = FORECAST_FEATURES + ["temp_c", "temp_roll_24h", "temp_roll_72h",
                                        "temp_max_today", "temp_min_today"]


def forecast_detector(df: pd.DataFrame, z_thresh: float = 4.0, n_splits: int = 6,
                      features=None) -> pd.DataFrame:
    """Predict 'normal' load out-of-sample (folds grouped by year) and score residuals.
    Default features are calendar-only; pass WEATHER_FEATURES to add temperature."""
    features = features or FORECAST_FEATURES
    n_splits = min(n_splits, df["year"].nunique())
    X, y = df[features], df["load_mw"]
    pred = pd.Series(np.nan, index=df.index)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=df["year"]):
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08,
                                              max_leaf_nodes=63, random_state=0)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = model.predict(X.iloc[te])
    resid = y - pred
    z = robust_z(resid)
    return pd.DataFrame({
        "fc_expected": pred,
        "fc_z": z,
        "fc_flag": z.abs() > z_thresh,
    })


# ---------------------------------------------------- 3. Isolation Forest
IF_FEATURES = ["load_mw", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
               "diff_1h", "diff_24h", "diff_168h", "roll_std_6h", "ratio_to_24h_mean"]


def isolation_forest_detector(df: pd.DataFrame, contamination: float = 0.01) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[IF_FEATURES])
    model = IsolationForest(n_estimators=300, contamination=contamination,
                            random_state=0, n_jobs=-1)
    model.fit(X)
    score = -model.score_samples(X)  # higher = more anomalous
    return pd.DataFrame({
        "if_score": score,
        "if_flag": model.predict(X) == -1,
    }, index=df.index)


# ---------------------------------------------------- 4. Autoencoder
def autoencoder_detector(df: pd.DataFrame, z_thresh: float = 3.5) -> pd.DataFrame:
    """Learn normal 24h load shapes; days that reconstruct badly are anomalous.
    Within a flagged day, the hours with the largest reconstruction error are flagged."""
    tmp = pd.DataFrame({"load": df["load_mw"].values,
                        "date": df.index.normalize(), "hour": df.index.hour})
    days = tmp.pivot_table(index="date", columns="hour", values="load").dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(days.values)
    ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), activation="relu", max_iter=400,
                      early_stopping=True, random_state=0, learning_rate_init=1e-3)
    ae.fit(X, X)
    err = (ae.predict(X) - X) ** 2                     # day x hour error matrix
    day_err = pd.Series(err.mean(axis=1), index=days.index)
    log_err = np.log(day_err)
    day_z = (log_err - log_err.median()) / (1.4826 * (log_err - log_err.median()).abs().median())
    day_flag = day_z > z_thresh

    hour_err = pd.DataFrame(err, index=days.index, columns=days.columns)
    hot_hours = hour_err.ge(0.5 * hour_err.max(axis=1), axis=0) & day_flag.values[:, None]

    key_date, key_hour = df.index.normalize(), df.index.hour
    stacked = hot_hours.stack()
    hour_flag = pd.Series(stacked.reindex(list(zip(key_date, key_hour)), fill_value=False).values,
                          index=df.index).astype(bool)
    return pd.DataFrame({
        "ae_day_error": day_err.reindex(key_date).values,
        "ae_day_z": day_z.reindex(key_date).values,
        "ae_day_flag": day_flag.reindex(key_date, fill_value=False).values.astype(bool),
        "ae_flag": hour_flag.values,
    }, index=df.index)


# ---------------------------------------------------- 5. Data-quality rules
def data_quality_detector(df: pd.DataFrame, min_run: int = 4, low_frac: float = 0.05) -> pd.DataFrame:
    """Deterministic checks that statistical models handle poorly:
    a stuck meter (>= min_run identical consecutive readings) and near-zero readings."""
    load = df["load_mw"]
    run_id = (load != load.shift()).cumsum()
    run_len = load.groupby(run_id).transform("size")
    flatline = run_len >= min_run
    baseline = load.rolling(24 * 30, center=True, min_periods=24).median()
    near_zero = load < low_frac * baseline
    return pd.DataFrame({"dq_flatline": flatline, "dq_near_zero": near_zero,
                         "dq_flag": flatline | near_zero}, index=df.index)


# ---------------------------------------------------- Ensemble
METHODS = ["stl", "fc", "if", "ae"]
METHOD_NAMES = {"stl": "STL + robust z", "fc": "Forecast residual (GBM)",
                "if": "Isolation Forest", "ae": "Autoencoder", "ens": "Ensemble (>=2 votes)",
                "dq": "Data-quality rules", "final": "Final (ensemble OR rules)"}


def run_all(df: pd.DataFrame, min_votes: int = 2, verbose: bool = True,
            cache_dir=None, tag: str = "real", use_weather: bool = False) -> pd.DataFrame:
    """Run all detectors. If cache_dir is given, each detector's output is saved and
    reused, so an interrupted run resumes where it stopped (delete the folder to refit).
    use_weather=True swaps in the temperature-aware forecast model."""
    if use_weather:
        def fc(d):
            return forecast_detector(d, features=WEATHER_FEATURES)
    else:
        fc = forecast_detector
    parts = []
    for name, fn in [("STL", stl_detector), ("Forecast", fc),
                     ("IsolationForest", isolation_forest_detector),
                     ("Autoencoder", autoencoder_detector)]:
        suffix = "_weather" if (use_weather and name == "Forecast") else ""
        cache = cache_dir / f"{tag}_{name}{suffix}.pkl" if cache_dir is not None else None
        if cache is not None and cache.exists():
            parts.append(pd.read_pickle(cache))
            continue
        if verbose:
            print(f"  running {name} ({tag}) ...", flush=True)
        part = fn(df)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            part.to_pickle(cache)
        parts.append(part)
    out = pd.concat([df[["load_mw"]]] + parts, axis=1)
    out["votes"] = out[[f"{m}_flag" for m in METHODS]].sum(axis=1)
    out["ens_flag"] = out["votes"] >= min_votes
    out = pd.concat([out, data_quality_detector(df)], axis=1)
    out["final_flag"] = out["ens_flag"] | out["dq_flag"]
    return out
