"""Build the offline interactive dashboard: outputs/dashboard.html

Run after run_pipeline.py and run_weather_experiment.py (it reuses their cached detector
outputs, so it takes a few seconds):   python src/build_dashboard.py
Then open outputs/dashboard.html in any browser - no internet or server needed.
"""
import json

import numpy as np
import pandas as pd

from data_prep import ROOT, add_features, prepare
from detectors import METHOD_NAMES, run_all
from weather import add_weather_features, load_temperature

OUT = ROOT / "outputs"
CACHE = OUT / "cache"
TEMPLATE = ROOT / "src" / "dashboard_template.html"
TARGET = OUT / "dashboard.html"

BITS = {"stl": 1, "fc": 2, "if": 4, "ae": 8, "dq": 16, "final": 32}
EVENTS = {
    "Hurricane Sandy": "2012-10-26",
    "Polar Vortex": "2014-01-04",
    "Blizzard Jonas": "2016-01-20",
    "July 2013 heat wave": "2013-07-13",
}


def flag_bits(res: pd.DataFrame) -> np.ndarray:
    bits = np.zeros(len(res), dtype=int)
    for m, b in BITS.items():
        bits |= res[f"{m}_flag"].astype(bool).values * b
    return bits


def main():
    clean_df, _ = prepare(save=False)
    df = add_features(clean_df)
    print("Loading detector results (cached) ...")
    res = run_all(df, cache_dir=CACHE, tag="real", verbose=False)

    temp = load_temperature()
    df_wx = add_weather_features(add_features(clean_df[["load_mw"]]), temp)
    res_wx = run_all(df_wx, cache_dir=CACHE, tag="wxperiod_real", use_weather=True, verbose=False)
    wx_off = int(res.index.get_loc(res_wx.index[0]))
    assert (res.index[wx_off: wx_off + len(res_wx)] == res_wx.index).all(), "weather period not contiguous"

    # metrics for the panels
    ev = pd.read_csv(OUT / "evaluation_metrics.csv")
    by_type = pd.read_csv(OUT / "evaluation_by_type.csv")
    wx_real = pd.read_csv(OUT / "weather_experiment" / "real_data_comparison.csv", index_col=0)
    wx_inj = pd.read_csv(OUT / "weather_experiment" / "injection_metrics_all_methods.csv")
    wx_final = wx_inj[wx_inj["method"] == "final"].set_index("variant")
    top = pd.read_csv(OUT / "top_anomalous_days.csv")

    type_table = by_type.pivot(index="method", columns="type", values="event_recall")
    order = ["stl", "fc", "if", "ae", "ens", "dq", "final"]
    type_table = type_table.loc[[m for m in order if m in type_table.index]]

    data = {
        "start": res.index[0].strftime("%Y-%m-%dT%H:%M:%S"),
        "n": len(res),
        "load": res["load_mw"].round().astype(int).tolist(),
        "exp": res["fc_expected"].round().astype(int).tolist(),
        "fl": flag_bits(res).tolist(),
        "wx": {
            "off": wx_off,
            "exp": res_wx["fc_expected"].round().astype(int).tolist(),
            "fl": flag_bits(res_wx).tolist(),
            "temp": df_wx.loc[res_wx.index, "temp_c"].round(1).tolist(),
        },
        "events": EVENTS,
        "kpi": {
            "hours": len(res),
            "final_flags": int(res["final_flag"].sum()),
            "f1_full": float(ev.loc[ev["method"] == METHOD_NAMES["final"], "point_f1"].iloc[0]),
            "er_full": float(ev.loc[ev["method"] == METHOD_NAMES["final"], "event_recall"].iloc[0]),
            "f1_cal": float(wx_final.loc["calendar", "point_f1"]),
            "f1_wx": float(wx_final.loc["weather", "point_f1"]),
            "er_cal": float(wx_final.loc["calendar", "event_recall"]),
            "er_wx": float(wx_final.loc["weather", "event_recall"]),
            "mape_cal": float(wx_real.loc["Calendar only", "MAPE_%"]),
            "mape_wx": float(wx_real.loc["Calendar + temperature", "MAPE_%"]),
            "flags_cal": int(wx_real.loc["Calendar only", "final_flags"]),
            "flags_wx": int(wx_real.loc["Calendar + temperature", "final_flags"]),
            "ext_cal": int(wx_real.loc["Calendar only", "final_flags_on_extreme_temp_days"]),
            "ext_wx": int(wx_real.loc["Calendar + temperature", "final_flags_on_extreme_temp_days"]),
            "prec_cal": float(wx_final.loc["calendar", "point_precision"]),
            "prec_wx": float(wx_final.loc["weather", "point_precision"]),
            "rec_cal": float(wx_final.loc["calendar", "point_recall"]),
            "rec_wx": float(wx_final.loc["weather", "point_recall"]),
        },
        "methods": {m: METHOD_NAMES[m] for m in order},
        "types": list(type_table.columns),
        "typeTable": {m: [round(float(x), 2) for x in type_table.loc[m]] for m in type_table.index},
        "pointTable": ev[["method", "point_precision", "point_recall", "point_f1", "event_recall"]]
        .round(3).values.tolist(),
        "top": top[["date", "anomalous_hours", "load_mean_mw", "load_min_mw", "load_max_mw", "max_votes"]]
        .head(20).values.tolist(),
    }
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, separators=(",", ":")))
    TARGET.write_text(html, encoding="utf-8")
    print(f"Dashboard written: {TARGET} ({TARGET.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
