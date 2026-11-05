"""Longitudinal infestation streams built from real recordings.

No public dataset follows one palm continuously across a real infestation, so
the time axis has to be constructed. This module does that in the most
defensible way available: by resampling *real* recordings from the dataset
rather than synthesising audio.

The model of early infestation
------------------------------
A young infestation means a small number of larvae inside a large trunk. Most
20-second recordings from such a tree therefore contain no feeding at all;
occasionally one catches a burst. As the larvae grow and multiply, the
*fraction* of recordings containing feeding rises.

So infestation is modelled as a rising **prevalence** p(t): the probability
that any given recording contains feeding activity. Each daily observation is
drawn from the tree's real clean recordings with probability 1 - p(t), and
from real infested recordings with probability p(t).

This is why single-recording classification struggles early and why
accumulating evidence helps: at p = 0.1 a per-clip detector sees nine
uninformative recordings for every informative one, while a sequential test
integrates all ten.
"""

from __future__ import annotations

import numpy as np


def logistic_growth(t, onset: float, ramp: float) -> np.ndarray:
    """Smooth 0 -> 1 population curve. `ramp` sets the time to go 10% -> 90%."""
    t = np.asarray(t, dtype=float)
    # 4.394 = 2 * ln(9), maps `ramp` onto the 10-90% rise time.
    return 1.0 / (1.0 + np.exp(-4.394 * (t - onset) / max(ramp, 1e-6)))


def linear_growth(t, onset: float, ramp: float) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return np.clip((t - onset) / max(ramp, 1e-6), 0.0, 1.0)


GROWTH = {"logistic": logistic_growth, "linear": linear_growth}


def prevalence_curve(n_obs: int, onset: int, ramp: float,
                     final_prevalence: float, growth: str = "logistic"):
    """p(t): probability that observation t contains feeding activity."""
    t = np.arange(n_obs, dtype=float)
    g = GROWTH[growth](t, float(onset), float(ramp))
    p = final_prevalence * g
    p[:onset] = 0.0
    return np.clip(p, 0.0, 1.0)


def make_stream(clean_scores, infested_scores, n_obs: int, onset: int,
                ramp: float, final_prevalence: float,
                growth: str = "logistic", rng=None):
    """One simulated monitoring stream for a single tree.

    Parameters
    ----------
    clean_scores, infested_scores : 1-D arrays of detector scores computed
        from that tree's real recordings (or, when a tree is observed in only
        one state, from held-out recordings -- state which was used).
    onset : observation index at which infestation begins.

    Returns
    -------
    scores : the observed stream
    p      : the prevalence curve that generated it
    """
    rng = np.random.default_rng() if rng is None else rng
    clean_scores = np.asarray(clean_scores, dtype=float).ravel()
    infested_scores = np.asarray(infested_scores, dtype=float).ravel()
    if clean_scores.size == 0 or infested_scores.size == 0:
        raise ValueError("Both clean and infested score pools must be non-empty.")

    p = prevalence_curve(n_obs, onset, ramp, final_prevalence, growth)
    draw_infested = rng.random(n_obs) < p
    out = np.where(
        draw_infested,
        rng.choice(infested_scores, size=n_obs, replace=True),
        rng.choice(clean_scores, size=n_obs, replace=True),
    )
    return out, p


def make_healthy_stream(clean_scores, n_obs: int, rng=None):
    """A stream from a tree that never becomes infested.

    Used to measure false alarms, which is what makes the comparison between
    detectors fair.
    """
    rng = np.random.default_rng() if rng is None else rng
    clean_scores = np.asarray(clean_scores, dtype=float).ravel()
    return rng.choice(clean_scores, size=n_obs, replace=True)
