"""Dataset scanning, manifest building and leakage-safe splitting.

The central design decision of this project lives here.

TreeVibes is organised as a set of recording folders, each one corresponding
to a device deployed on a particular tree over a particular period. Every
folder therefore shares a tree, a sensor, a mounting and a background noise
environment.

If clips are split randomly into train and test, clips from the same folder
appear on both sides. A classifier can then reach near-perfect scores by
recognising the *recording conditions* rather than the insect. Published
results on this dataset report accuracies at or near 100%, and at least one
paper remarks that the data seem "too simple" for deep learning -- which is
what leakage looks like from the inside.

`grouped_splits` assigns whole folders to folds, so every evaluation answers
the question that matters in the field: does this work on a tree the model
has never heard?
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import LABELS, RANDOM_SEED, N_SPLITS

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif"}

# Folder/file name fragments that indicate each class. Ordered so that the
# more specific "not infested" style labels are tested before "infested".
CLEAN_PATTERNS = [
    r"\bclean\b", r"\bhealthy\b", r"\bnot[_\- ]?infest", r"\bnon[_\- ]?infest",
    r"\bno[_\- ]?infest", r"\bnegative\b", r"\bnormal\b", r"\bcontrol\b",
]
INFESTED_PATTERNS = [
    r"\binfest", r"\bpest\b", r"\bweevil\b", r"\brpw\b", r"\bpositive\b",
    r"\bborer\b", r"\blarva",
]


def _match(text: str, patterns) -> bool:
    return any(re.search(p, text) for p in patterns)


def infer_label(path: Path, root: Path):
    """Infer clean/infested from any component of the relative path."""
    rel = path.relative_to(root).as_posix().lower().replace("-", "_")
    # Check clean first: "not_infested" contains "infest".
    if _match(rel, CLEAN_PATTERNS):
        return "clean"
    if _match(rel, INFESTED_PATTERNS):
        return "infested"
    return None


def infer_group(path: Path, root: Path, label: str | None) -> str:
    """Identify the recording session / tree that a clip belongs to.

    Uses the deepest directory that is not merely the class label, so that
    layouts like  <root>/<tree_id>/<label>/clip.wav  and
                  <root>/<label>/<tree_id>/clip.wav
    both resolve to the tree rather than to the class.
    """
    parts = list(path.relative_to(root).parts[:-1])
    cleaned = [
        p for p in parts
        if not _match(p.lower().replace("-", "_"), CLEAN_PATTERNS + INFESTED_PATTERNS)
    ]
    if cleaned:
        return "/".join(cleaned)
    # Fall back to the class folder itself, then to the file stem, so that
    # every clip still receives *some* group and nothing is silently merged.
    return parts[0] if parts else path.stem


def scan_dataset(root) -> pd.DataFrame:
    """Walk a dataset directory and build a manifest of labelled clips."""
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    rows, skipped = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_EXT:
            continue
        label = infer_label(path, root)
        if label is None:
            skipped.append(path)
            continue
        rows.append({
            "path": str(path),
            "label": label,
            "y": LABELS[label],
            "group": infer_group(path, root, label),
        })

    if not rows:
        listing = sorted({p.relative_to(root).parts[0]
                          for p in root.rglob("*") if p.is_file()})[:20]
        raise ValueError(
            "No labelled audio files found.\n"
            f"Looked under: {root}\n"
            f"Top-level entries seen: {listing}\n"
            "Expected folder or file names containing 'clean'/'healthy' or "
            "'infested'. If this dataset uses a different convention, edit "
            "CLEAN_PATTERNS / INFESTED_PATTERNS in src/data.py."
        )

    df = pd.DataFrame(rows)
    if skipped:
        print(f"  note: {len(skipped)} audio files had no recognisable label "
              f"and were skipped (e.g. {Path(skipped[0]).name})")
    return df


def describe(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().to_dict()
    n_groups = df["group"].nunique()
    per_group = df.groupby("group")["label"].nunique()
    mixed = int((per_group > 1).sum())
    lines = [
        f"  clips              : {len(df)}",
        f"  class balance      : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"  recording groups   : {n_groups}",
        f"  groups with both classes : {mixed}",
    ]
    sizes = Counter(df["group"]).most_common()
    lines.append(f"  largest group      : {sizes[0][0]} ({sizes[0][1]} clips)")
    return "\n".join(lines)


def subsample(df: pd.DataFrame, max_per_class: int | None,
              seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Cap the number of clips per class while preserving group structure.

    TreeVibes is large and heavily skewed (far more infested clips than
    clean). Processing all of it takes hours on a laptop, and the extra
    majority-class clips add little. Sampling is done *within* each recording
    group in proportion to that group's size, so no tree is dropped entirely
    and the grouped evaluation stays meaningful.
    """
    if not max_per_class:
        return df

    rng = np.random.RandomState(seed)
    kept = []
    for label, sub in df.groupby("label", sort=True):
        if len(sub) <= max_per_class:
            kept.append(sub)
            continue
        groups = sub["group"].unique()
        # Even share per group, then top up randomly to hit the target.
        per_group = max(1, max_per_class // max(1, len(groups)))
        picks = []
        for g, gsub in sub.groupby("group", sort=True):
            n = min(len(gsub), per_group)
            picks.append(gsub.sample(n=n, random_state=rng))
        picked = pd.concat(picks)
        if len(picked) > max_per_class:
            picked = picked.sample(n=max_per_class, random_state=rng)
        elif len(picked) < max_per_class:
            remaining = sub.drop(index=picked.index)
            extra = min(max_per_class - len(picked), len(remaining))
            if extra > 0:
                picked = pd.concat(
                    [picked, remaining.sample(n=extra, random_state=rng)])
        kept.append(picked)

    out = pd.concat(kept).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    return out


def grouped_splits(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """Stratified K-fold that never splits a recording group across folds.

    Yields (train_index, test_index) over the rows of `df`.
    """
    n_groups = df["group"].nunique()
    if n_groups < 2:
        raise ValueError(
            f"Only {n_groups} recording group found. Grouped evaluation needs "
            "at least 2 so that test trees are unseen during training."
        )
    n_splits = int(min(n_splits, n_groups))
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    return splitter.split(df, df["y"].values, groups=df["group"].values)


def random_splits(df: pd.DataFrame, n_splits: int = N_SPLITS):
    """Leakage-prone clip-level split, kept ONLY as a comparison baseline.

    This reproduces the evaluation protocol that yields near-perfect published
    numbers. It is reported alongside the grouped result to quantify how much
    of that performance is an artefact of the split.
    """
    from sklearn.model_selection import StratifiedKFold
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED
    )
    return splitter.split(df, df["y"].values)
