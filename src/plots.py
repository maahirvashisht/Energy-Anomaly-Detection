"""All figures for the project (saved as PNG in outputs/figures)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detectors import METHOD_NAMES

BASE = "#4a5a6a"
ACCENT = "#d1495b"
SOFT = "#9fb3c8"
COLORS = {"stl": "#2e86ab", "fc": "#f18f01", "if": "#6a994e", "ae": "#7b2cbf", "ens": "#d1495b",
          "dq": "#8d99ae", "final": "#1b1b1e"}

plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def eda_full_series(df, path):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(df.index, df["load_mw"], color=SOFT, lw=0.3, label="Hourly load")
    ax.plot(df.index, df["load_mw"].rolling(24 * 7, center=True).mean(), color=BASE, lw=1.2,
            label="7-day rolling mean")
    ax.set(title="PJM East hourly electricity load, 2002-2018", ylabel="MW")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, path)


def eda_heatmap(df, path):
    pivot = df.pivot_table(index="dayofweek", columns="hour", values="load_mw", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.imshow(pivot.values / 1000, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2))
    ax.set(xlabel="Hour of day", title="Average load by hour and weekday (GW)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="GW")
    _save(fig, path)


def eda_monthly(df, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    data = [df.loc[df["month"] == m, "load_mw"].values / 1000 for m in range(1, 13)]
    axes[0].boxplot(data, showfliers=False)
    axes[0].set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    axes[0].set(title="Load distribution by month", ylabel="GW")
    for season, months, c in [("Winter", [12, 1, 2], "#2e86ab"), ("Summer", [6, 7, 8], ACCENT),
                              ("Spring/Autumn", [3, 4, 5, 9, 10, 11], "#6a994e")]:
        prof = df[df["month"].isin(months)].groupby("hour")["load_mw"].mean() / 1000
        axes[1].plot(prof.index, prof.values, label=season, color=c, lw=2)
    axes[1].set(title="Average daily profile by season", xlabel="Hour", ylabel="GW")
    axes[1].legend(frameon=False)
    _save(fig, path)


def detection_overview(res, path):
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(res.index, res["load_mw"], color=SOFT, lw=0.3)
    a = res[res["final_flag"]]
    ax.scatter(a.index, a["load_mw"], s=6, color=ACCENT, zorder=3,
               label=f"Detected anomaly ({len(a):,} hours)")
    ax.set(title="Detected anomalies across the full history", ylabel="MW")
    ax.legend(loc="upper left", frameon=False)
    _save(fig, path)


def zoom_event(res, start, end, title, path):
    r = res.loc[start:end]
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]})
    ax = axes[0]
    ax.plot(r.index, r["load_mw"], color=BASE, lw=1.3, label="Actual load")
    ax.plot(r.index, r["fc_expected"], color="#f18f01", lw=1, ls="--",
            label="Expected (calendar model)")
    a = r[r["final_flag"]]
    ax.scatter(a.index, a["load_mw"], color=ACCENT, s=18, zorder=3, label="Detected anomaly")
    ax.set(title=title, ylabel="MW")
    ax.legend(frameon=False, loc="lower right")
    ax2 = axes[1]
    for i, m in enumerate(["stl", "fc", "if", "ae"]):
        f = r[r[f"{m}_flag"]]
        ax2.scatter(f.index, np.full(len(f), i), marker="|", s=120, color=COLORS[m])
    ax2.set_yticks(range(4), [METHOD_NAMES[m] for m in ["stl", "fc", "if", "ae"]])
    ax2.set_ylim(-0.6, 3.6)
    ax2.set_title("Which detector fired", fontsize=10)
    _save(fig, path)


def injection_example(res_inj, clean_load, events, start, end, path):
    r = res_inj.loc[start:end]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(clean_load.loc[start:end].index, clean_load.loc[start:end], color=SOFT, lw=1,
            label="Original")
    ax.plot(r.index, r["load_mw"], color=BASE, lw=1, label="With injected anomalies")
    ev = events[(events["start"] >= start) & (events["start"] <= end)]
    for _, e in ev.iterrows():
        ax.axvspan(e["start"] - pd.Timedelta(hours=1), e["end"] + pd.Timedelta(hours=1),
                   color="#f6c85f", alpha=0.45)
        ax.text(e["start"], ax.get_ylim()[1] * 0.98, e["type"], fontsize=8, va="top")
    a = r[r["final_flag"]]
    ax.scatter(a.index, a["load_mw"], color=ACCENT, s=16, zorder=3, label="Detection (final)")
    ax.set(title="Synthetic anomaly injection (yellow = injected event)", ylabel="MW")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def method_comparison(ev_scores, path):
    piv = ev_scores.pivot(index="type", columns="method", values="event_recall")
    order = ["stl", "fc", "if", "ae", "ens", "final"]
    piv = piv[order]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    w = 0.13
    x = np.arange(len(piv.index))
    for i, m in enumerate(order):
        ax.bar(x + (i - 2.5) * w, piv[m], w, label=METHOD_NAMES[m], color=COLORS[m])
    ax.set_xticks(x, piv.index)
    ax.set(ylim=(0, 1.05), ylabel="Share of injected events caught",
           title="Event recall by anomaly type and method")
    ax.legend(frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, path)
