"""Data loading, cleaning and feature engineering for PJM East hourly load."""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "PJME_hourly.csv"
PROCESSED_PATH = ROOT / "data" / "processed" / "pjme_clean.csv"


def load_raw(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df.rename(columns={"Datetime": "datetime", "PJME_MW": "load_mw"})


def _longest_gap(mask: pd.Series) -> int:
    """Length of the longest run of True values."""
    runs = (mask != mask.shift()).cumsum()
    lengths = mask.groupby(runs).sum()
    return int(lengths.max()) if len(lengths) else 0


def clean(df: pd.DataFrame):
    """Sort, merge duplicate timestamps, reindex to a regular hourly grid
    and interpolate the missing hours. Returns (clean_df, report)."""
    report = {
        "raw_rows": len(df),
        "duplicate_timestamps": int(df["datetime"].duplicated().sum()),
    }
    # Duplicates come from daylight-saving changeovers -> average them.
    s = df.groupby("datetime")["load_mw"].mean().sort_index()
    full_index = pd.date_range(s.index.min(), s.index.max(), freq="h")
    report["missing_hours"] = int(len(full_index) - len(s))
    s = s.reindex(full_index)
    imputed = s.isna()
    report["longest_gap_hours"] = _longest_gap(imputed)
    s = s.interpolate(method="time")

    out = pd.DataFrame({"load_mw": s, "imputed": imputed})
    out.index.name = "datetime"
    report["clean_rows"] = len(out)
    report["start"] = str(out.index.min())
    report["end"] = str(out.index.max())
    return out, report


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar, cyclical and change-based features."""
    df = df.copy()
    idx = df.index
    df["hour"] = idx.hour
    df["dayofweek"] = idx.dayofweek
    df["month"] = idx.month
    df["year"] = idx.year
    df["dayofyear"] = idx.dayofyear
    holidays = USFederalHolidayCalendar().holidays(idx.min(), idx.max())
    df["is_holiday"] = idx.normalize().isin(holidays).astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    for name, period, col in [("hour", 24, "hour"),
                              ("dow", 7, "dayofweek"),
                              ("doy", 365.25, "dayofyear")]:
        df[f"{name}_sin"] = np.sin(2 * np.pi * df[col] / period)
        df[f"{name}_cos"] = np.cos(2 * np.pi * df[col] / period)

    load = df["load_mw"]
    df["diff_1h"] = load.diff(1)
    df["diff_24h"] = load.diff(24)
    df["diff_168h"] = load.diff(168)
    df["roll_std_6h"] = load.rolling(6, min_periods=2).std()
    df["ratio_to_24h_mean"] = load / load.rolling(24, min_periods=6).mean()
    change_cols = ["diff_1h", "diff_24h", "diff_168h", "roll_std_6h", "ratio_to_24h_mean"]
    df[change_cols] = df[change_cols].bfill()
    return df


def prepare(path=RAW_PATH, save=True):
    raw = load_raw(path)
    clean_df, report = clean(raw)
    if save:
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_csv(PROCESSED_PATH)
    return clean_df, report


if __name__ == "__main__":
    _, rep = prepare()
    for k, v in rep.items():
        print(f"{k:>22}: {v}")
