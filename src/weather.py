"""Hourly temperature for Philadelphia (largest city in the PJM East zone).

Source: Kaggle "Historical Hourly Weather Data 2012-2017" (OpenWeatherMap), file
temperature.csv, downloaded from a GitHub mirror. Values are Kelvin with UTC timestamps.
Verified as UTC: the coolest summer hour is 10:00 and the warmest is 19:00 in the raw data,
i.e. 06:00 and 15:00 US Eastern time.
"""
import pandas as pd

from data_prep import ROOT

TEMP_PATH = ROOT / "data" / "raw" / "historical_hourly_temperature.csv"
CITY = "Philadelphia"


def load_temperature(path=TEMP_PATH, city=CITY) -> pd.Series:
    t = pd.read_csv(path, usecols=["datetime", city], parse_dates=["datetime"])
    s = (t.set_index("datetime")[city] - 273.15).rename("temp_c")
    # UTC -> US Eastern local time (PJM load timestamps are local), drop tz info
    s.index = s.index.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)
    s = s.groupby(level=0).mean()  # DST fall-back creates duplicate local hours
    full = pd.date_range(s.index.min().ceil("h"), s.index.max().floor("h"), freq="h")
    return s.reindex(full).interpolate(method="time", limit=6)


def add_weather_features(df: pd.DataFrame, temp: pd.Series) -> pd.DataFrame:
    """Join temperature onto the load frame and add smoothed / daily features.
    Rows without temperature are dropped."""
    df = df.join(temp, how="inner")
    df = df[df["temp_c"].notna()].copy()
    df["temp_roll_24h"] = df["temp_c"].rolling(24, min_periods=6).mean()
    df["temp_roll_72h"] = df["temp_c"].rolling(72, min_periods=12).mean()
    day = df.index.normalize()
    df["temp_max_today"] = df.groupby(day)["temp_c"].transform("max")
    df["temp_min_today"] = df.groupby(day)["temp_c"].transform("min")
    cols = ["temp_roll_24h", "temp_roll_72h"]
    df[cols] = df[cols].bfill()
    return df
