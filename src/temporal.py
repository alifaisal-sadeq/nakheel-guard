"""Real-time, temperature-aware monitoring — the second contribution.

Why this exists
---------------
Wood-boring larvae are ectotherms. Their feeding activity rises steeply with
trunk temperature, so the *acoustic* evidence of infestation is not constant:
a given infested tree sounds quiet in winter and busy in summer.

That creates a confound which single-recording classifiers cannot see, because
they never observe the same tree twice. A detector that monitors one tree over
months walks straight into it: as spring arrives, a perfectly healthy tree
gets warmer, its background activity rises, and a temperature-blind detector
reports an infestation.

The 2026 systematic review of RPW detection states the gap directly -- no
system it appraised demonstrated prospectively validated environmental
adaptation of thresholds (gap RG-6).

TreeVibes encodes temperature in every filename, so the correction is
available for free.

Method
------
`TemperatureBaseline` models a tree's healthy score as a function of
temperature rather than as a single number:

    score ~ f(temperature) + residual

f is fitted on the tree's own learning period by robust local regression
(binned medians with linear interpolation, which needs no assumption about
the shape of the response). Detection then operates on the *residual*: how
far is this tree from where it should be **at this temperature**, not from
its annual average.

`seasonal_streams` builds monitoring streams from genuine timestamps rather
than simulated ordering, so the sequential detector can be evaluated on the
real temporal structure of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# temperature-conditioned baseline
# --------------------------------------------------------------------------

@dataclass
class TemperatureBaseline:
    """A tree's healthy acoustic state as a function of temperature.

    Compare with `sequential.TreeBaseline`, which models the healthy state as
    a single centre and scale. That is correct only if activity is
    temperature-independent, which for an ectotherm it is not.
    """
    edges: np.ndarray = field(default=None)
    centres: np.ndarray = field(default=None)
    scales: np.ndarray = field(default=None)
    global_centre: float = 0.0
    global_scale: float = 1.0
    n_obs: int = 0
    n_bins: int = 0

    @classmethod
    def fit(cls, scores, temperatures, n_bins: int = 6,
            min_per_bin: int = 8) -> "TemperatureBaseline":
        """Fit from a tree's own healthy learning period.

        Bins are quantile-based so each holds a similar number of
        observations. Bins that stay too small are merged into the global
        estimate, so a sparse temperature range degrades gracefully rather
        than producing a wild local fit.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        temps = np.asarray(temperatures, dtype=float).ravel()
        ok = np.isfinite(scores) & np.isfinite(temps)
        scores, temps = scores[ok], temps[ok]

        if scores.size < 10:
            raise ValueError(
                f"Temperature baseline needs at least 10 observations, "
                f"got {scores.size}."
            )

        g_centre = float(np.median(scores))
        g_scale = max(1.4826 * float(np.median(np.abs(scores - g_centre))), 1e-6)

        n_bins = int(max(1, min(n_bins, scores.size // min_per_bin)))
        if n_bins <= 1:
            return cls(edges=None, centres=None, scales=None,
                       global_centre=g_centre, global_scale=g_scale,
                       n_obs=int(scores.size), n_bins=0)

        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(temps, qs))
        if edges.size < 3:
            return cls(edges=None, centres=None, scales=None,
                       global_centre=g_centre, global_scale=g_scale,
                       n_obs=int(scores.size), n_bins=0)

        idx = np.clip(np.digitize(temps, edges[1:-1]), 0, edges.size - 2)
        centres = np.full(edges.size - 1, g_centre)
        scales = np.full(edges.size - 1, g_scale)
        for b in range(edges.size - 1):
            m = idx == b
            if m.sum() >= min_per_bin:
                c = float(np.median(scores[m]))
                s = 1.4826 * float(np.median(np.abs(scores[m] - c)))
                centres[b] = c
                scales[b] = max(s, 0.25 * g_scale)   # floor avoids a tight bin

        return cls(edges=edges, centres=centres, scales=scales,
                   global_centre=g_centre, global_scale=g_scale,
                   n_obs=int(scores.size), n_bins=int(edges.size - 1))

    def _interp(self, temps):
        """Expected centre and scale at each temperature.

        Bin midpoints are interpolated linearly, and held flat outside the
        fitted range rather than extrapolated -- extrapolating a biological
        response curve beyond the observed temperatures is not defensible.
        """
        if self.edges is None:
            return (np.full(temps.shape, self.global_centre),
                    np.full(temps.shape, self.global_scale))
        mids = 0.5 * (self.edges[:-1] + self.edges[1:])
        c = np.interp(temps, mids, self.centres,
                      left=self.centres[0], right=self.centres[-1])
        s = np.interp(temps, mids, self.scales,
                      left=self.scales[0], right=self.scales[-1])
        return c, np.maximum(s, 1e-6)

    def residual(self, scores, temperatures) -> np.ndarray:
        """Standardised deviation from the expected value *at this temperature*."""
        scores = np.asarray(scores, dtype=float).ravel()
        temps = np.asarray(temperatures, dtype=float).ravel()
        temps = np.where(np.isfinite(temps), temps, np.nanmedian(temps))
        c, s = self._interp(temps)
        return (scores - c) / s

    def naive_residual(self, scores) -> np.ndarray:
        """The temperature-blind comparison: deviation from the annual average.

        This is what a detector does when it ignores the environment, and it
        is the baseline the temperature-aware version is measured against.
        """
        scores = np.asarray(scores, dtype=float).ravel()
        return (scores - self.global_centre) / self.global_scale

    def curve(self, lo=None, hi=None, n=100):
        """The fitted response, for plotting."""
        if self.edges is None:
            t = np.linspace(lo if lo is not None else 0,
                            hi if hi is not None else 30, n)
            return t, np.full(n, self.global_centre)
        lo = self.edges[0] if lo is None else lo
        hi = self.edges[-1] if hi is None else hi
        t = np.linspace(lo, hi, n)
        c, _ = self._interp(t)
        return t, c


# --------------------------------------------------------------------------
# real-time streams
# --------------------------------------------------------------------------

def daily_aggregate(df: pd.DataFrame, score_col: str = "score",
                    how: str = "median") -> pd.DataFrame:
    """One row per tree-day: the natural unit for longitudinal monitoring.

    Devices record several times a day at different temperatures. Collapsing
    to the day evens out the diurnal cycle and matches how a farm would
    actually consume alerts.
    """
    d = df.dropna(subset=["timestamp"]).copy()
    d["date"] = pd.to_datetime(d["timestamp"]).dt.floor("D")
    agg = {score_col: how}
    if "temperature_c" in d.columns:
        agg["temperature_c"] = "mean"
    out = (d.groupby(["group", "date"], as_index=False)
             .agg({**agg, "path": "count"})
             .rename(columns={"path": "n_recordings"}))
    return out.sort_values(["group", "date"]).reset_index(drop=True)


def seasonal_streams(daily: pd.DataFrame, min_days: int = 30) -> dict:
    """Split a daily table into one real-time stream per tree."""
    streams = {}
    for g, sub in daily.groupby("group"):
        if len(sub) >= min_days:
            streams[g] = sub.sort_values("date").reset_index(drop=True)
    return streams


def split_learning_period(stream: pd.DataFrame, fraction: float = 0.4,
                          min_days: int = 20):
    """Divide a tree's stream into its learning period and monitoring period.

    A deployed device models its own tree during an initial window, then
    monitors. Splitting by time (never at random) is what makes the
    evaluation causal: nothing from the future informs the baseline.
    """
    n_fit = max(min_days, int(fraction * len(stream)))
    n_fit = min(n_fit, len(stream) - 5)
    if n_fit < min_days:
        return None, None
    return stream.iloc[:n_fit].copy(), stream.iloc[n_fit:].copy()


# --------------------------------------------------------------------------
# the confound demonstration
# --------------------------------------------------------------------------

def temperature_confound_test(streams: dict, labels: dict,
                              fraction: float = 0.4) -> pd.DataFrame:
    """Measure how badly a temperature-blind detector drifts on healthy trees.

    For every tree: fit both baselines on the learning period, then compare
    the mean residual over the monitoring period. On a *healthy* tree the
    residual should stay near zero. If the naive residual drifts upward while
    the temperature-aware one does not, the drift is seasonal warming being
    misread as infestation.
    """
    rows = []
    for g, stream in streams.items():
        fit, monitor = split_learning_period(stream, fraction)
        if fit is None or "temperature_c" not in stream.columns:
            continue
        try:
            bl = TemperatureBaseline.fit(fit["score"].values,
                                         fit["temperature_c"].values)
        except ValueError:
            continue

        naive = bl.naive_residual(monitor["score"].values)
        aware = bl.residual(monitor["score"].values,
                            monitor["temperature_c"].values)
        rows.append({
            "group": g,
            "label": labels.get(g, "unknown"),
            "n_fit_days": len(fit),
            "n_monitor_days": len(monitor),
            "fit_temp_mean": float(fit["temperature_c"].mean()),
            "monitor_temp_mean": float(monitor["temperature_c"].mean()),
            "temp_shift": float(monitor["temperature_c"].mean()
                                - fit["temperature_c"].mean()),
            "naive_residual_mean": float(np.mean(naive)),
            "aware_residual_mean": float(np.mean(aware)),
            "naive_residual_sd": float(np.std(naive)),
            "aware_residual_sd": float(np.std(aware)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# bridge: features -> scored, time-stamped records
# --------------------------------------------------------------------------

def score_recordings(manifest: pd.DataFrame, X, clip_ids, w, scaler,
                     tree_baselines: dict | None = None) -> pd.DataFrame:
    """Attach a detector score to every recording in a manifest.

    Window-level features are averaged back up to the recording they came
    from, because the monitoring unit is a recording, not a 2-second window.

    `tree_baselines` maps group -> sequential.TreeBaseline. When supplied,
    scores are expressed relative to each tree's own baseline; otherwise the
    raw projection onto the shift direction is returned.
    """
    import numpy as np

    Xs = scaler.transform(np.asarray(X, dtype=float))
    proj = Xs @ w

    by_clip = pd.DataFrame({"path": np.asarray(clip_ids), "proj": proj})
    per_clip = by_clip.groupby("path", as_index=False)["proj"].mean()

    out = manifest.merge(per_clip, on="path", how="inner")
    if out.empty:
        raise ValueError(
            "No manifest row matched a feature row. The feature cache was "
            "probably built from a different manifest -- re-extract."
        )

    if tree_baselines:
        scores = np.full(len(out), np.nan)
        for g, bl in tree_baselines.items():
            m = (out["group"] == g).values
            if m.any():
                scores[m] = (out.loc[m, "proj"].values - bl.centre) / bl.scale
        out["score"] = scores
        out["score"] = out["score"].fillna(out["proj"])
    else:
        out["score"] = out["proj"]
    return out
