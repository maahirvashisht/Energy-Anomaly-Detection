"""End-to-end pipeline: python src/run_pipeline.py  (run from the project folder)."""
import json
import time

import pandas as pd

import plots
from data_prep import ROOT, add_features, prepare
from detectors import METHOD_NAMES, METHODS, run_all
from evaluate import inject_anomalies, score_methods

OUT = ROOT / "outputs"
FIG = OUT / "figures"
TEST_START, TEST_END = "2017-01-01", "2017-12-31 23:00"
REAL_EVENTS = {  # known real-world events used as a sanity check
    "Hurricane Sandy": ("2012-10-28", "2012-11-02"),
    "Polar Vortex": ("2014-01-05", "2014-01-09"),
    "Blizzard Jonas": ("2016-01-22", "2016-01-25"),
}


def main():
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)

    print("1/5 Cleaning data")
    clean_df, report = prepare()
    df = add_features(clean_df)
    print(json.dumps(report, indent=2))

    print("2/5 EDA figures")
    plots.eda_full_series(df, FIG / "01_full_series.png")
    plots.eda_heatmap(df, FIG / "02_hour_weekday_heatmap.png")
    plots.eda_monthly(df, FIG / "03_monthly_and_seasonal_profiles.png")

    print("3/5 Detecting anomalies on real data")
    res = run_all(df, cache_dir=OUT / "cache", tag="real")
    res["imputed"] = clean_df["imputed"]
    res.to_csv(OUT / "anomalies_hourly.csv")

    summary = {m: int(res[f"{m}_flag"].sum()) for m in METHODS + ["ens", "dq", "final"]}
    daily = (res.assign(date=res.index.date)
                .groupby("date")
                .agg(load_mean_mw=("load_mw", "mean"), load_min_mw=("load_mw", "min"),
                     load_max_mw=("load_mw", "max"), anomalous_hours=("final_flag", "sum"),
                     max_votes=("votes", "max"), max_abs_fc_z=("fc_z", lambda z: z.abs().max())))
    top_days = daily[daily["anomalous_hours"] > 0].sort_values(
        ["anomalous_hours", "max_abs_fc_z"], ascending=False).head(25)
    top_days.round(2).to_csv(OUT / "top_anomalous_days.csv")

    real_check = []
    for name, (a, b) in REAL_EVENTS.items():
        r = res.loc[a:b]
        real_check.append({"event": name, "window": f"{a} to {b}",
                           **{METHOD_NAMES[m]: int(r[f"{m}_flag"].sum()) for m in METHODS + ["ens", "final"]}})
        plots.zoom_event(res, a, b, f"{name}: detected anomalies",
                         FIG / f"05_event_{name.lower().replace(' ', '_')}.png")
    real_check = pd.DataFrame(real_check)
    real_check.to_csv(OUT / "real_event_check.csv", index=False)
    plots.detection_overview(res, FIG / "04_detection_overview.png")

    print("4/5 Evaluating with injected anomalies (re-running all detectors)")
    inj_load, label, events = inject_anomalies(df["load_mw"], TEST_START, TEST_END)
    inj_df = add_features(pd.DataFrame({"load_mw": inj_load}))
    res_inj = run_all(inj_df, cache_dir=OUT / "cache", tag="injected")
    point, by_type = score_methods(res_inj, label, events, TEST_START, TEST_END)
    point["method"] = point["method"].map(METHOD_NAMES)
    events.to_csv(OUT / "injected_events.csv", index=False)
    point.round(4).to_csv(OUT / "evaluation_metrics.csv", index=False)
    by_type.assign(method_name=by_type["method"].map(METHOD_NAMES)).round(4).to_csv(
        OUT / "evaluation_by_type.csv", index=False)
    plots.method_comparison(by_type, FIG / "07_method_comparison.png")
    first = events["start"].iloc[0]
    plots.injection_example(res_inj, df["load_mw"], events, first - pd.Timedelta(days=2),
                            first + pd.Timedelta(days=26), FIG / "06_injection_example.png")

    print("5/5 Summary")
    print("Flagged hours per method:", summary)
    print(real_check.to_string(index=False))
    print(point.round(4).to_string(index=False))
    print(by_type.pivot(index="type", columns="method", values="event_recall").round(2))
    print("Top anomalous days:\n", top_days.head(10).round(1))
    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({"data_report": report, "flagged_hours": summary}, f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
