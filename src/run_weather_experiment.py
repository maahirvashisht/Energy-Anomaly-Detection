"""Experiment: does adding temperature improve anomaly detection?

Compares two versions of the pipeline on the period where weather data exists
(Oct 2012 - Nov 2017):
  A. calendar-only forecast model (as in run_pipeline.py)
  B. forecast model with Philadelphia temperature features
Everything else (STL, Isolation Forest, autoencoder, rules) is identical.

Run from the project folder:  python src/run_weather_experiment.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plots
from data_prep import ROOT, add_features, prepare
from detectors import run_all
from evaluate import inject_anomalies, score_methods
from weather import add_weather_features, load_temperature

OUT = ROOT / "outputs" / "weather_experiment"
FIG = OUT / "figures"
CACHE = ROOT / "outputs" / "cache"
TEST_START, TEST_END = "2017-01-01", "2017-11-28 23:00"
EVENTS = {
    "Hurricane Sandy": ("2012-10-28", "2012-11-02"),
    "Polar Vortex": ("2014-01-05", "2014-01-09"),
    "Blizzard Jonas": ("2016-01-22", "2016-01-25"),
    "July 2013 heat wave": ("2013-07-14", "2013-07-20"),
}
VARIANTS = {"calendar": False, "weather": True}
LABELS = {"calendar": "Calendar only", "weather": "Calendar + temperature"}


def build_frame(load: pd.Series, temp: pd.Series) -> pd.DataFrame:
    return add_weather_features(add_features(pd.DataFrame({"load_mw": load})), temp)


def forecast_accuracy(res):
    err = res["load_mw"] - res["fc_expected"]
    return {"MAPE_%": float((err.abs() / res["load_mw"]).mean() * 100),
            "RMSE_MW": float(np.sqrt((err ** 2).mean()))}


def plot_load_vs_temp(df, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    s = df.sample(min(len(df), 15000), random_state=0)
    ax.scatter(s["temp_c"], s["load_mw"] / 1000, s=2, alpha=0.25, color=plots.BASE)
    ax.set(xlabel="Philadelphia temperature (°C)", ylabel="PJM East load (GW)",
           title="Load vs temperature: heating and cooling demand")
    plots._save(fig, path)


def plot_expected_comparison(res_cal, res_wx, start, end, title, path):
    a, b = res_cal.loc[start:end], res_wx.loc[start:end]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    ax.plot(a.index, a["load_mw"], color=plots.BASE, lw=1.4, label="Actual load")
    ax.plot(a.index, a["fc_expected"], color="#f18f01", lw=1, ls="--", label="Expected: calendar only")
    ax.plot(b.index, b["fc_expected"], color="#2e86ab", lw=1.2, ls="--",
            label="Expected: calendar + temperature")
    for res, mk, c, lab in [(a, "x", "#f18f01", "Flag: calendar"), (b, "o", "#2e86ab", "Flag: weather-aware")]:
        f = res[res["final_flag"]]
        ax.scatter(f.index, f["load_mw"], marker=mk, color=c, s=28, zorder=3, label=lab)
    ax.set(title=title, ylabel="MW")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    axes[1].plot(b.index, b["temp_c"], color=plots.ACCENT, lw=1.2)
    axes[1].set_ylabel("Temp °C")
    plots._save(fig, path)


def plot_metric_bars(table, path):
    metrics = ["point_precision", "point_recall", "point_f1", "event_recall"]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(metrics))
    for i, (v, c) in enumerate([("calendar", "#f18f01"), ("weather", "#2e86ab")]):
        vals = table.loc[v, metrics].values
        ax.bar(x + (i - 0.5) * 0.36, vals, 0.36, color=c, label=LABELS[v])
        for xi, val in zip(x, vals):
            ax.text(xi + (i - 0.5) * 0.36, val + 0.01, f"{val:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, ["Precision", "Recall", "F1", "Event recall"])
    ax.set(ylim=(0, 1.1), title="Final detector on injected anomalies (2017)")
    ax.legend(frameon=False, loc="lower right")
    plots._save(fig, path)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    clean_df, _ = prepare(save=False)
    temp = load_temperature()
    df = build_frame(clean_df["load_mw"], temp)
    print(f"Overlap period: {df.index.min()} to {df.index.max()} ({len(df):,} hours)")
    print(f"Correlation load vs temp: {df['load_mw'].corr(df['temp_c']):.2f} (U-shaped, so linear corr is weak)")
    plot_load_vs_temp(df, FIG / "w1_load_vs_temperature.png")

    real, inj = {}, {}
    inj_load, label, events = inject_anomalies(df["load_mw"], TEST_START, TEST_END)
    inj_df = build_frame(inj_load, temp).loc[df.index]
    for v, use_wx in VARIANTS.items():
        print(f"Running pipeline: {LABELS[v]}")
        real[v] = run_all(df, cache_dir=CACHE, tag="wxperiod_real", use_weather=use_wx)
        real[v]["temp_c"] = df["temp_c"]
        inj[v] = run_all(inj_df, cache_dir=CACHE, tag="wxperiod_inj", use_weather=use_wx)

    # 1. forecast accuracy  2. flags overall / on extreme-temperature days / in real events
    daily_t = df["temp_c"].groupby(df.index.normalize()).mean()
    lo, hi = daily_t.quantile([0.05, 0.95])
    extreme_days = daily_t[(daily_t < lo) | (daily_t > hi)].index
    extreme_mask = df.index.normalize().isin(extreme_days)
    rows = []
    for v in VARIANTS:
        r = real[v]
        row = {"variant": LABELS[v], **forecast_accuracy(r),
               "forecast_flags": int(r["fc_flag"].sum()),
               "final_flags": int(r["final_flag"].sum()),
               "final_flags_on_extreme_temp_days": int(r.loc[extreme_mask, "final_flag"].sum())}
        for name, (a, b) in EVENTS.items():
            row[f"flags_{name}"] = int(r.loc[a:b, "final_flag"].sum())
        rows.append(row)
    real_table = pd.DataFrame(rows).set_index("variant")

    # 3. injection benchmark
    score_rows, type_rows = [], []
    for v in VARIANTS:
        point, by_type = score_methods(inj[v], label, events, TEST_START, TEST_END)
        point.insert(0, "variant", v)
        by_type.insert(0, "variant", v)
        score_rows.append(point)
        type_rows.append(by_type)
    scores = pd.concat(score_rows)
    by_type = pd.concat(type_rows)
    final = scores[scores["method"] == "final"].set_index("variant")
    fc_only = scores[scores["method"] == "fc"].set_index("variant")

    real_table.round(4).to_csv(OUT / "real_data_comparison.csv")
    scores.round(4).to_csv(OUT / "injection_metrics_all_methods.csv", index=False)
    by_type.round(4).to_csv(OUT / "injection_recall_by_type.csv", index=False)

    plot_metric_bars(final, FIG / "w2_injection_metrics.png")
    for name, (a, b) in EVENTS.items():
        plot_expected_comparison(real["calendar"], real["weather"], a, b,
                                 f"{name}: calendar-only vs weather-aware",
                                 FIG / f"w3_{name.lower().replace(' ', '_')}.png")

    pd.set_option("display.width", 200)
    print("\nREAL DATA\n", real_table.round(2).T)
    print("\nINJECTION - final detector\n",
          final[["flagged_hours", "point_precision", "point_recall", "point_f1", "event_recall"]].round(4))
    print("\nINJECTION - forecast detector alone\n",
          fc_only[["flagged_hours", "point_precision", "point_recall", "point_f1", "event_recall"]].round(4))
    print("\nEvent recall by type (final)\n",
          by_type[by_type["method"] == "final"].pivot(index="type", columns="variant",
                                                      values="event_recall"))
    with open(OUT / "experiment_info.json", "w", encoding="utf-8") as f:
        json.dump({"overlap_start": str(df.index.min()), "overlap_end": str(df.index.max()),
                   "extreme_temp_thresholds_c": [float(lo), float(hi)]}, f, indent=2)


if __name__ == "__main__":
    main()
