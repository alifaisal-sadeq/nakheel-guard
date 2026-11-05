"""The headline experiment: how much earlier does sequential detection alert?

Both detectors are calibrated to the *same* false-alarm budget on healthy
trees before they are compared. Without that step the comparison is empty,
because any detector looks faster when it is allowed to cry wolf more often.

What is measured
----------------
detection delay  - observations between infestation onset and the alarm.
detection rate   - fraction of infested streams detected at all within the
                   observation window.
false alarms     - measured on healthy streams, to confirm the calibration
                   actually held.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sequential import (
    cusum, per_clip_alarm, calibrate_threshold, calibrate_per_clip_threshold,
    TreeBaseline,
)
from simulate import make_stream, make_healthy_stream


def build_score_pools(X, y, groups, test_trees, w, scaler,
                      baseline_fraction: float = 0.5, seed: int = 0):
    """Score held-out recordings against per-tree baselines.

    A deployed device is installed on a tree that is healthy at the time, so
    a baseline always comes from *clean* recordings of the tree being
    monitored. Half of each test tree's clean recordings are used to fit its
    baseline; the rest are scored as the healthy pool.

    The infested pool is the harder case. If the monitored tree also has
    infested recordings, those are used and the comparison is entirely
    within-tree -- the ideal. If it does not (common: a tree in these
    datasets is usually recorded in one state only), infested recordings from
    *other* held-out trees are scored against this tree's baseline. That
    remains a valid simulation of "this tree starts sounding infested", but
    the measured shift then also contains between-tree variation. The
    returned `info` records which case applied, and that must be stated in
    any result.
    """
    rng = np.random.default_rng(seed)
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    test_trees = list(test_trees)

    clean_pool, infested_pool = [], []
    within_tree, cross_tree, skipped = [], [], []

    # Infested recordings available among the held-out trees, for the fallback.
    held = np.isin(groups, test_trees)
    foreign_infested = X[held & (y == 1)]

    for t in test_trees:
        m = groups == t
        Xc, Xi = X[m & (y == 0)], X[m & (y == 1)]
        if Xc.shape[0] < 6:
            skipped.append(str(t))          # no usable healthy baseline
            continue
        idx = rng.permutation(Xc.shape[0])
        n_fit = max(3, int(baseline_fraction * Xc.shape[0]))
        fit_idx, score_idx = idx[:n_fit], idx[n_fit:]
        if score_idx.size == 0:
            skipped.append(str(t))
            continue

        bl = TreeBaseline.fit(Xc[fit_idx], w, scaler)
        clean_pool.append(bl.score(Xc[score_idx], w, scaler))

        if Xi.shape[0] >= 3:
            infested_pool.append(bl.score(Xi, w, scaler))
            within_tree.append(str(t))
        elif foreign_infested.shape[0] >= 3:
            others = X[held & (y == 1) & ~m]
            if others.shape[0] >= 3:
                infested_pool.append(bl.score(others, w, scaler))
                cross_tree.append(str(t))

    if not clean_pool:
        raise ValueError(
            "No held-out tree had enough clean recordings to form a baseline. "
            "Choose test trees that include clean-labelled trees."
        )
    if not infested_pool:
        raise ValueError(
            "No infested recordings available for the held-out trees. "
            "Include at least one infested tree in the test set."
        )

    info = {
        "trees_within_tree_contrast": within_tree,
        "trees_cross_tree_fallback": cross_tree,
        "trees_skipped_no_baseline": skipped,
        "contrast": "within-tree" if cross_tree == [] else "cross-tree fallback",
    }
    return np.concatenate(clean_pool), np.concatenate(infested_pool), info


def run_comparison(clean_scores, infested_scores, *,
                   prevalences=(0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
                   n_obs: int = 180, onset: int = 60, ramp: float = 45.0,
                   n_trials: int = 300,
                   false_alarms_per_year: float = 0.5,
                   cusum_k: float = 0.5,
                   consecutive: int = 2,
                   seed: int = 0) -> pd.DataFrame:
    """Compare sequential CUSUM against per-clip classification.

    `clean_scores` / `infested_scores` are detector scores from real
    recordings, produced by TreeBaseline.score().
    """
    rng = np.random.default_rng(seed)

    # --- calibrate both detectors on healthy streams, same budget ----------
    healthy_pool = [make_healthy_stream(clean_scores, n_obs, rng)
                    for _ in range(200)]
    h = calibrate_threshold(
        healthy_pool, k=cusum_k,
        target_false_alarms_per_year=false_alarms_per_year,
        obs_per_year=n_obs, rng_seed=seed)
    thr = calibrate_per_clip_threshold(
        healthy_pool, consecutive=consecutive,
        target_false_alarms_per_year=false_alarms_per_year,
        obs_per_year=n_obs, rng_seed=seed)

    # --- verify the calibration on fresh healthy streams -------------------
    fa_cusum = fa_clip = 0
    for _ in range(n_trials):
        s = make_healthy_stream(clean_scores, n_obs, rng)
        fa_cusum += cusum(s, k=cusum_k, h=h)[0] is not None
        fa_clip += per_clip_alarm(s, thr, consecutive) is not None

    rows = []
    for p in prevalences:
        d_cusum, d_clip = [], []
        n_det_cusum = n_det_clip = 0
        for _ in range(n_trials):
            s, _ = make_stream(clean_scores, infested_scores, n_obs, onset,
                               ramp, p, rng=rng)
            a1, _ = cusum(s, k=cusum_k, h=h)
            a2 = per_clip_alarm(s, thr, consecutive)
            # Only alarms at or after onset count as detections; an alarm
            # before onset is a false alarm, not an early detection.
            if a1 is not None and a1 >= onset:
                n_det_cusum += 1
                d_cusum.append(a1 - onset)
            if a2 is not None and a2 >= onset:
                n_det_clip += 1
                d_clip.append(a2 - onset)

        rows.append({
            "final_prevalence": p,
            "cusum_detection_rate": n_det_cusum / n_trials,
            "cusum_median_delay": float(np.median(d_cusum)) if d_cusum else np.nan,
            "cusum_mean_delay": float(np.mean(d_cusum)) if d_cusum else np.nan,
            "perclip_detection_rate": n_det_clip / n_trials,
            "perclip_median_delay": float(np.median(d_clip)) if d_clip else np.nan,
            "perclip_mean_delay": float(np.mean(d_clip)) if d_clip else np.nan,
        })

    df = pd.DataFrame(rows)
    df.attrs["cusum_threshold"] = h
    df.attrs["perclip_threshold"] = thr
    df.attrs["consecutive"] = consecutive
    df.attrs["observed_false_alarm_rate_cusum"] = fa_cusum / n_trials
    df.attrs["observed_false_alarm_rate_perclip"] = fa_clip / n_trials
    df.attrs["false_alarm_budget"] = false_alarms_per_year
    df.attrs["n_obs"] = n_obs
    df.attrs["onset"] = onset
    df.attrs["n_trials"] = n_trials
    return df


def summarise(df: pd.DataFrame) -> str:
    a = df.attrs
    lines = [
        "Calibration (both detectors held to the same budget)",
        f"  budget                      : {a['false_alarm_budget']:.2f} false alarms "
        f"per tree per {a['n_obs']} observations",
        f"  observed, sequential CUSUM  : {a['observed_false_alarm_rate_cusum']:.3f}",
        f"  observed, per-clip rule     : {a['observed_false_alarm_rate_perclip']:.3f}",
        f"  CUSUM threshold h           : {a['cusum_threshold']:.2f}",
        f"  per-clip threshold          : {a['perclip_threshold']:.2f} "
        f"({a['consecutive']} consecutive)",
        "",
        "Detection (delay in observations after onset)",
    ]
    hdr = (f"  {'prevalence':>10} | {'CUSUM rate':>10} {'delay':>7} | "
           f"{'clip rate':>10} {'delay':>7}")
    lines += [hdr, "  " + "-" * (len(hdr) - 2)]
    for _, r in df.iterrows():
        cd = "  n/a" if np.isnan(r.cusum_median_delay) else f"{r.cusum_median_delay:5.0f}"
        pd_ = "  n/a" if np.isnan(r.perclip_median_delay) else f"{r.perclip_median_delay:5.0f}"
        lines.append(
            f"  {r.final_prevalence:>10.2f} | {r.cusum_detection_rate:>10.2f} {cd:>7} | "
            f"{r.perclip_detection_rate:>10.2f} {pd_:>7}")
    return "\n".join(lines)


def check_direction_transfer(X, y, groups, test_trees, w, scaler,
                             seed: int = 0, verbose: bool = True) -> dict:
    """Does the learned shift direction survive on unseen trees?

    A discriminant direction can separate the training trees perfectly and
    still be useless -- or worse than useless -- on a new tree. When every
    tree is wholly clean or wholly infested, "which tree" and "which class"
    are the same variable, so the direction is free to encode tree
    idiosyncrasies that point somewhere arbitrary on a tree it has not seen.

    This measures the effect size on training trees and on held-out trees and
    classifies the outcome:

      transfers   - held-out d is positive and a reasonable share of training d
      collapses   - held-out d near zero: the direction carries nothing general
      inverts     - held-out d negative: it points the WRONG WAY on new trees

    "inverts" is the dangerous one, and it is silent without this check. A
    sequential detector built on an inverted direction accumulates evidence
    downward, so it becomes *less* likely to alarm the worse the infestation
    gets -- failing exactly when it matters, while looking well-behaved on
    every training diagnostic.
    """
    X, y, groups = np.asarray(X), np.asarray(y), np.asarray(groups)
    test_mask = np.isin(groups, list(test_trees))

    def _d(Xsub, ysub):
        if len(np.unique(ysub)) < 2:
            return float("nan")
        proj = scaler.transform(Xsub) @ w
        a, b = proj[ysub == 1], proj[ysub == 0]
        if a.size < 2 or b.size < 2:
            return float("nan")
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
        return float((a.mean() - b.mean()) / pooled)

    d_train = _d(X[~test_mask], y[~test_mask])
    d_test = _d(X[test_mask], y[test_mask])

    if not np.isfinite(d_test):
        verdict = "untestable"
    elif d_test < -0.2:
        verdict = "inverts"
    elif d_test < 0.2:
        verdict = "collapses"
    elif d_test < 0.4 * abs(d_train):
        verdict = "weak"
    else:
        verdict = "transfers"

    out = {"d_train": d_train, "d_test": d_test,
           "retained_fraction": float(d_test / d_train) if d_train else float("nan"),
           "verdict": verdict}

    if verbose:
        print(f"  direction transfer: train d={d_train:+.2f}  "
              f"held-out d={d_test:+.2f}  ->  {verdict.upper()}")
        if verdict == "inverts":
            print("  The direction points the WRONG WAY on unseen trees.")
            print("  Any sequential result built on it is meaningless, and")
            print("  this is the clearest evidence that what the model learned")
            print("  was tree identity rather than infestation.")
        elif verdict == "collapses":
            print("  The direction carries nothing that generalises to a new tree.")
    return out
