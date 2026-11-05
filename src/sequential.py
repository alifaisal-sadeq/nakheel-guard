"""Sequential, per-tree change detection — the core contribution.

Motivation
----------
Every published acoustic method for this pest classifies a single recording:
"does this 20-second clip sound infested?" That framing suits a technician
walking a plantation with a probe. It does not suit a sensor bolted to one
palm that will record the same tree every day for months.

A fixed sensor makes a different question available, and an easier one:
*has this tree changed from its own baseline?*

Method
------
The approach separates two things that the single-clip framing conflates.

  tree identity   - where a tree sits in feature space. Differs enormously
                    between palms (size, moisture, density, mounting, local
                    noise). This is what leaks into random train/test splits
                    and inflates published accuracy.

  infestation shift - the *direction* in which feeding activity moves a tree's
                    features. Shared across trees, because the underlying
                    biology is the same.

So we estimate them separately:

1. `fit_shift_direction` centres every training tree on its own clean mean,
   which deletes tree identity, then learns a single discriminant direction w
   from the residuals. w is the shared "what infestation looks like" axis.

2. `TreeBaseline` is fitted per deployed tree from its own quiet period. It
   needs no labels — only the assumption that the tree is healthy when the
   device is installed.

3. Each new observation is projected onto w after subtracting that tree's own
   baseline, giving a scalar evidence score.

4. `cusum` accumulates those scores over time. A small persistent shift that
   no single clip can resolve becomes detectable once enough of it has piled
   up. Detectable effect size falls roughly as 1/sqrt(N) in the number of
   observations, which is the formal reason this detects earlier.

5. `calibrate_threshold` sets the alarm level by simulation on healthy-only
   streams, so the operating point is stated as false alarms per tree-year
   rather than as an accuracy number.

Reference for the sequential test: E. S. Page (1954), "Continuous inspection
schemes", Biometrika 41(1/2), 100-115.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler


# --------------------------------------------------------------------------
# 1. shared shift direction, learned with tree identity removed
# --------------------------------------------------------------------------

def fit_shift_direction(X, y, groups, shrinkage: bool = True):
    """Learn the direction in feature space along which infestation moves a tree.

    Every group (tree) is centred on its own *clean* mean before the
    discriminant is fitted, so the direction cannot encode which tree a
    sample came from. Trees with no clean samples are dropped, because
    without a clean reference their offset cannot be removed.

    Returns
    -------
    w       : unit vector, the shift direction
    scaler  : the StandardScaler used, to apply to future data
    info    : dict with diagnostics
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    residuals, labels, used, dropped = [], [], [], []
    for g in np.unique(groups):
        m = groups == g
        clean = m & (y == 0)
        if clean.sum() < 2:
            dropped.append(str(g))
            continue
        mu = Xs[clean].mean(axis=0)
        residuals.append(Xs[m] - mu)      # tree identity removed
        labels.append(y[m])
        used.append(str(g))

    if not residuals:
        raise ValueError(
            "No tree has enough clean recordings to establish a baseline.\n"
            "The shift direction needs trees observed in both states, or at "
            "least trees with a clean reference period."
        )

    R = np.vstack(residuals)
    L = np.concatenate(labels)
    if len(np.unique(L)) < 2:
        raise ValueError(
            "After centring, only one class remains. This happens when every "
            "tree is wholly clean or wholly infested, so no within-tree "
            "contrast exists. Use fit_shift_direction_pooled() instead and "
            "state the limitation."
        )

    # Fisher discriminant direction: Sigma^-1 (mu_1 - mu_0), with shrinkage so
    # it stays stable when features outnumber samples.
    mu0 = R[L == 0].mean(axis=0)
    mu1 = R[L == 1].mean(axis=0)
    delta = mu1 - mu0

    if shrinkage:
        cov = LedoitWolf(assume_centered=False).fit(R).covariance_
        w = np.linalg.solve(cov + 1e-8 * np.eye(cov.shape[0]), delta)
    else:
        w = delta

    norm = np.linalg.norm(w)
    if norm < 1e-12:
        raise ValueError("Degenerate shift direction (zero norm).")
    w = w / norm
    w = _orient(w, R, L)      # a direction is defined only up to sign

    info = {
        "n_trees_used": len(used),
        "trees_used": used,
        "trees_dropped_no_clean_reference": dropped,
        "n_residual_samples": int(R.shape[0]),
        "within_tree_contrast": True,
        "effect_size_cohens_d": float(
            (w @ mu1 - w @ mu0) / (np.std(R @ w) + 1e-12)),
    }
    return w, scaler, info


def fit_shift_direction_pooled(X, y, groups, shrinkage: bool = True):
    """Fallback when no tree is observed in both states.

    A warning about identifiability
    -------------------------------
    When every tree is wholly clean or wholly infested, "which tree" and
    "which class" are the *same variable*. No procedure can separate the
    infestation shift from tree identity, because nothing in the data
    distinguishes them. This is a property of the experimental design, not
    something a better estimator can fix.

    An earlier version of this function centred each tree on its own mean --
    the same trick that works when trees are observed in both states. Under
    single-state trees that is actively harmful: centring a wholly-infested
    tree on its own mean subtracts the infestation signal along with the tree
    identity, leaving class means that differ only by noise. The resulting
    direction was arbitrary, and on test data it pointed the *wrong way*
    (Cohen's d around -1.0 instead of +1.0), so the sequential detector got
    steadily worse as infestation rose.

    So this version does NOT centre per tree. It fits a plain regularised
    discriminant on standardised features and states honestly that the
    direction is confounded with tree identity. Results obtained this way
    describe what is learnable from between-tree differences, which is
    exactly the quantity notebook 02 shows to be inflated by leakage.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    groups = np.asarray(groups)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    mu0, mu1 = Xs[y == 0].mean(axis=0), Xs[y == 1].mean(axis=0)
    delta = mu1 - mu0
    if shrinkage:
        cov = LedoitWolf().fit(Xs).covariance_
        w = np.linalg.solve(cov + 1e-8 * np.eye(cov.shape[0]), delta)
    else:
        w = delta
    w = w / (np.linalg.norm(w) + 1e-12)
    w = _orient(w, Xs, y)

    info = {
        "n_trees_used": int(len(np.unique(groups))),
        "within_tree_contrast": False,
        "effect_size_cohens_d": _cohens_d(Xs @ w, y),
        "warning": ("No tree observed in both states, so tree identity and "
                    "class are perfectly confounded. The direction cannot be "
                    "separated from between-tree variation; report this."),
    }
    return w, scaler, info


def _cohens_d(proj, y) -> float:
    a, b = proj[y == 1], proj[y == 0]
    if a.size < 2 or b.size < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-12
    return float((a.mean() - b.mean()) / pooled)


def _orient(w, Xs, y):
    """Force the direction to point from clean toward infested.

    A discriminant direction is only defined up to sign, and several routes
    to it (shrinkage, solving, degenerate class means) can return the
    negative. An inverted direction is silently catastrophic for a sequential
    detector: evidence of infestation accumulates downward, so the detector
    becomes *less* likely to alarm the worse the infestation gets.

    This check costs nothing and makes that failure impossible.
    """
    proj = Xs @ w
    if proj[y == 1].mean() < proj[y == 0].mean():
        return -w
    return w


# --------------------------------------------------------------------------
# 2. per-tree baseline, fitted on that tree's own quiet period
# --------------------------------------------------------------------------

@dataclass
class TreeBaseline:
    """Statistical description of one tree while it is healthy.

    Fitted without labels from the device's learning period after install.
    Robust statistics (median, MAD) are used so that a few stray events
    during the learning period do not distort the baseline.
    """
    mu: np.ndarray = field(default=None)
    centre: float = 0.0
    scale: float = 1.0
    n_obs: int = 0

    @classmethod
    def fit(cls, X_clean, w, scaler) -> "TreeBaseline":
        X_clean = np.asarray(X_clean, dtype=np.float64)
        if X_clean.shape[0] < 3:
            raise ValueError(
                f"Baseline needs at least 3 observations, got {X_clean.shape[0]}."
            )
        Xs = scaler.transform(X_clean)
        mu = np.median(Xs, axis=0)
        proj = (Xs - mu) @ w
        centre = float(np.median(proj))
        mad = float(np.median(np.abs(proj - centre)))
        # 1.4826 * MAD estimates the standard deviation for Gaussian data.
        scale = max(1.4826 * mad, 1e-6)
        return cls(mu=mu, centre=centre, scale=scale, n_obs=int(X_clean.shape[0]))

    def score(self, X, w, scaler) -> np.ndarray:
        """Standardised evidence score, one per observation.

        Zero means "indistinguishable from this tree's healthy state".
        Positive means "shifted along the infestation direction".
        """
        Xs = scaler.transform(np.atleast_2d(np.asarray(X, dtype=np.float64)))
        proj = (Xs - self.mu) @ w
        return (proj - self.centre) / self.scale


# --------------------------------------------------------------------------
# 3. sequential test
# --------------------------------------------------------------------------

def cusum(scores, k: float = 0.5, h: float = 5.0):
    """One-sided CUSUM. Returns (alarm_index or None, running statistic).

    k is the reference value: shifts smaller than k are treated as noise and
    bled off, so the statistic does not drift upward on a healthy tree.
    h is the alarm threshold, set by `calibrate_threshold`.
    """
    scores = np.asarray(scores, dtype=np.float64)
    S = np.zeros(scores.size + 1)
    alarm = None
    for t, s in enumerate(scores):
        S[t + 1] = max(0.0, S[t] + s - k)
        if alarm is None and S[t + 1] > h:
            alarm = t
    return alarm, S[1:]


def calibrate_threshold(baseline_scores_iter, k: float = 0.5,
                        target_false_alarms_per_year: float = 0.5,
                        obs_per_year: int = 365,
                        h_grid=None, rng_seed: int = 0):
    """Choose the alarm threshold h from healthy-only streams.

    The operating point is expressed the way a farm manager would state it:
    how often may a healthy tree raise a false alarm? Everything else follows.

    `baseline_scores_iter` supplies 1-D arrays of scores from healthy periods.
    """
    pools = [np.asarray(s, dtype=np.float64).ravel()
             for s in baseline_scores_iter]
    pools = [p for p in pools if p.size > 0]
    if not pools:
        raise ValueError("No healthy score streams supplied for calibration.")
    pool = np.concatenate(pools)

    if h_grid is None:
        h_grid = np.arange(1.0, 30.01, 0.25)

    rng = np.random.default_rng(rng_seed)
    n_runs, run_len = 400, obs_per_year
    # Bootstrap healthy years by resampling observed healthy scores.
    sims = rng.choice(pool, size=(n_runs, run_len), replace=True)

    best_h = float(h_grid[-1])
    for h in h_grid:
        alarms = 0
        for row in sims:
            a, _ = cusum(row, k=k, h=h)
            if a is not None:
                alarms += 1
        rate = alarms / n_runs          # false alarms per tree-year
        if rate <= target_false_alarms_per_year:
            best_h = float(h)
            break
    return best_h


# --------------------------------------------------------------------------
# 4. the comparison baseline: what the published framing would do
# --------------------------------------------------------------------------

def per_clip_alarm(scores, threshold: float, consecutive: int = 1):
    """Single-clip decision rule, optionally requiring consecutive positives.

    This represents the published approach applied to a monitoring stream:
    classify each recording independently, alert when one (or `consecutive`
    in a row) crosses a fixed threshold. It has no memory beyond that window,
    which is precisely what the sequential test adds.
    """
    scores = np.asarray(scores, dtype=np.float64)
    run = 0
    for t, s in enumerate(scores):
        run = run + 1 if s > threshold else 0
        if run >= consecutive:
            return t
    return None


def calibrate_per_clip_threshold(baseline_scores_iter, consecutive: int = 1,
                                 target_false_alarms_per_year: float = 0.5,
                                 obs_per_year: int = 365, rng_seed: int = 0):
    """Match the per-clip rule to the same false-alarm budget as CUSUM.

    Without this the comparison would be meaningless: any detector looks
    faster if it is allowed to cry wolf more often.
    """
    pools = [np.asarray(s, dtype=np.float64).ravel()
             for s in baseline_scores_iter]
    pool = np.concatenate([p for p in pools if p.size > 0])

    rng = np.random.default_rng(rng_seed)
    n_runs, run_len = 400, obs_per_year
    sims = rng.choice(pool, size=(n_runs, run_len), replace=True)

    lo, hi = float(np.min(pool)), float(np.max(pool) + 10.0)
    for _ in range(50):                       # bisection on the threshold
        mid = (lo + hi) / 2
        alarms = sum(per_clip_alarm(row, mid, consecutive) is not None
                     for row in sims)
        rate = alarms / n_runs
        if rate > target_false_alarms_per_year:
            lo = mid
        else:
            hi = mid
    return hi
