"""Evaluation: the real data has no anomaly labels, so we inject synthetic anomalies
with known location and type into a test year, re-run every detector, and measure
how many injected events each method catches."""
import numpy as np
import pandas as pd

from detectors import METHODS

ANOMALY_TYPES = {
    # name: (min_len, max_len) in hours
    "spike": (1, 1),          # meter spike: one hour 40-80% too high
    "dip": (1, 1),            # one hour 50-70% too low
    "level_shift": (6, 24),   # sustained +-20-35% for several hours
    "flatline": (12, 48),     # stuck meter: value frozen
    "dropout": (2, 6),        # outage / data loss: near-zero reading
}


def inject_anomalies(load: pd.Series, start: str, end: str, n_per_type: int = 10,
                     seed: int = 42, margin: int = 72):
    rng = np.random.default_rng(seed)
    s = load.copy()
    label = pd.Series(0, index=s.index)
    window_pos = np.where((s.index >= start) & (s.index <= end))[0]
    taken = np.zeros(len(s), dtype=bool)
    events = []

    for a_type, (lo, hi) in ANOMALY_TYPES.items():
        placed = 0
        while placed < n_per_type:
            length = int(rng.integers(lo, hi + 1))
            p0 = int(rng.choice(window_pos[:-hi - margin]))
            if taken[max(0, p0 - margin): p0 + length + margin].any():
                continue
            seg = slice(p0, p0 + length)
            base = s.iloc[seg].values
            if a_type == "spike":
                new = base * rng.uniform(1.4, 1.8)
            elif a_type == "dip":
                new = base * rng.uniform(0.3, 0.5)
            elif a_type == "level_shift":
                factor = rng.choice([rng.uniform(1.2, 1.35), rng.uniform(0.65, 0.8)])
                new = base * factor
            elif a_type == "flatline":
                new = np.full(length, s.iloc[p0])
            else:  # dropout
                new = rng.uniform(0, 500, size=length)
            s.iloc[seg] = new
            label.iloc[seg] = 1
            taken[seg] = True
            events.append({"type": a_type, "start": s.index[p0],
                           "end": s.index[p0 + length - 1], "length_h": length})
            placed += 1
    return s, label, pd.DataFrame(events).sort_values("start").reset_index(drop=True)


def score_methods(results: pd.DataFrame, label: pd.Series, events: pd.DataFrame,
                  start: str, end: str, tolerance_h: int = 2):
    """Point-level precision/recall/F1 and event-level recall per anomaly type.
    An event counts as detected if any flag falls inside it (+- tolerance)."""
    win = (results.index >= start) & (results.index <= end)
    y = label[win].astype(bool)
    rows, ev_rows = [], []
    for m in METHODS + ["ens", "dq", "final"]:
        flag = results.loc[win, f"{m}_flag"].astype(bool)
        tp = int((flag & y).sum())
        fp = int((flag & ~y).sum())
        fn = int((~flag & y).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

        full_flag = results[f"{m}_flag"].astype(bool)
        hit = []
        for _, e in events.iterrows():
            lo = e["start"] - pd.Timedelta(hours=tolerance_h)
            hi = e["end"] + pd.Timedelta(hours=tolerance_h)
            hit.append(bool(full_flag.loc[lo:hi].any()))
        hit = pd.Series(hit, index=events.index)
        rows.append({"method": m, "flagged_hours": int(flag.sum()), "point_precision": prec,
                     "point_recall": rec, "point_f1": f1, "event_recall": hit.mean()})
        for t, grp in events.groupby("type"):
            ev_rows.append({"method": m, "type": t, "event_recall": hit[grp.index].mean()})
    return pd.DataFrame(rows), pd.DataFrame(ev_rows)
