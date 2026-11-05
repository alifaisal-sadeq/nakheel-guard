"""Turn a manifest of audio clips into a cached feature matrix.

Each source clip is cut into overlapping analysis windows; every window
becomes one training row but inherits the clip's label *and* the clip's
recording group, so that windows from one clip can never be split across a
train/test boundary either.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from config import SAMPLE_RATE, RESULTS_DIR
from features import load_audio, frame_windows, window_features


def _process_clip(args):
    path, label, y_lab, group = args
    try:
        audio = load_audio(path)
        windows = frame_windows(audio)
        rows = [window_features(w) for w in windows]
        return path, label, y_lab, group, rows, None
    except Exception as exc:  # keep one bad file from killing the run
        return path, label, y_lab, group, [], f"{type(exc).__name__}: {exc}"


def build_matrix(df: pd.DataFrame, workers: int | None = None, quiet: bool = False):
    """Return (X, y, groups, clip_ids, feature_names) for a manifest."""
    tasks = [
        (r.path, r.label, r.y, r.group)
        for r in df.itertuples(index=False)
    ]
    feats, labels, groups, clip_ids, errors = [], [], [], [], []

    workers = workers or min(8, max(1, (len(tasks) + 3) // 4))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_clip, t): t for t in tasks}
        for fut in as_completed(futures):
            path, label, y_lab, group, rows, err = fut.result()
            done += 1
            if err:
                errors.append((path, err))
                continue
            for row in rows:
                feats.append(row)
                labels.append(y_lab)
                groups.append(group)
                clip_ids.append(path)
            if not quiet and done % 200 == 0:
                print(f"    {done}/{len(tasks)} clips processed", flush=True)

    if not feats:
        raise RuntimeError(
            "No windows were extracted. Every clip failed or was too short.\n"
            + ("\n".join(f"  {Path(p).name}: {e}" for p, e in errors[:5])
               if errors else "")
        )

    names = sorted(feats[0].keys())
    X = np.array([[f.get(n, 0.0) for n in names] for f in feats], dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if errors and not quiet:
        print(f"  warning: {len(errors)} clip(s) failed to process")
        for p, e in errors[:3]:
            print(f"    {Path(p).name}: {e}")

    return X, np.array(labels), np.array(groups), np.array(clip_ids), names


def cache_path(tag: str) -> Path:
    return RESULTS_DIR / f"features_{tag}.npz"


def save_cache(tag, X, y, groups, clip_ids, names):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(tag), X=X, y=y, groups=groups,
        clip_ids=clip_ids, names=np.array(names),
    )


def load_cache(tag):
    p = cache_path(tag)
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    return d["X"], d["y"], d["groups"], d["clip_ids"], list(d["names"])


# --------------------------------------------------------------------------
# sanity checks
# --------------------------------------------------------------------------

def check_features(X, names, y=None, verbose: bool = True) -> dict:
    """Flag degenerate features before they reach a model.

    Written after a real failure: the burst detector's threshold was set so
    high it never fired, so every burst_* column was a constant zero. The
    pipeline ran, the model trained, and the accuracy looked plausible --
    nothing announced that a whole family of features was dead.

    A constant feature is not a harmless zero. It is a silent claim that
    something was measured when it was not.
    """
    X = np.asarray(X, dtype=float)
    names = list(names)

    sd = X.std(axis=0)
    constant = [names[i] for i in np.where(sd < 1e-12)[0]]
    near_const = [names[i] for i in np.where((sd >= 1e-12) & (sd < 1e-6))[0]]
    nonfinite = [names[i] for i in np.where(~np.isfinite(X).all(axis=0))[0]]

    families: dict[str, list] = {}
    for n in names:
        fam = n.split("_")[0]
        families.setdefault(fam, []).append(n)
    dead_families = [
        fam for fam, cols in families.items()
        if all(c in constant for c in cols)
    ]

    report = {
        "n_features": len(names),
        "constant": constant,
        "near_constant": near_const,
        "nonfinite": nonfinite,
        "dead_families": dead_families,
        "ok": not constant and not nonfinite,
    }

    if verbose:
        if report["ok"]:
            print(f"  feature check: OK ({len(names)} features, none degenerate)")
        else:
            print(f"  feature check: {len(constant)} constant, "
                  f"{len(near_const)} near-constant, {len(nonfinite)} non-finite")
            if dead_families:
                print(f"  DEAD FEATURE FAMILIES: {dead_families}")
                print("  A whole family of zeros means that measurement is not "
                      "working. Fix it before reporting anything.")
            elif constant:
                print(f"  constant: {constant[:8]}"
                      f"{' ...' if len(constant) > 8 else ''}")
    return report
