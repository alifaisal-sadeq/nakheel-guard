"""Loader for the real TreeVibes dataset.

TreeVibes (Rigakis et al., 2021) is the only public dataset of in-trunk
vibration recordings with infestation labels. It is what Boulila et al. (2023)
and Karar et al. (2022) evaluate on, so it is the right benchmark for
comparison -- with one caveat that must be stated in any result:

    TreeVibes records *Xylotrechus chinensis* (tiger longicorn beetle),
    mostly in mulberry trees. It is NOT Red Palm Weevil. The 2026 systematic
    review of RPW detection calls it "a cross-species wood-borer proxy rather
    than an RPW-native benchmark". Treat every result here as evidence about
    wood-borer detection methodology, not about RPW specifically.

What this module adds over the generic scanner in `data.py`
-----------------------------------------------------------
TreeVibes filenames encode acquisition metadata:

    F_20200222172127_4_T14.0
    ^ ^              ^ ^
    | |              | temperature, degrees C
    | |              index of the file within its folder
    | timestamp, yyyymmddhhmmss
    prefix

Two consequences, both of which matter for this project:

1. **Real time.** Recordings span about six months. Ordering a tree's files by
   timestamp produces a genuine monitoring stream, so the sequential detector
   can be evaluated on real temporal data instead of simulated ordering.

2. **Temperature.** Insect activity is strongly temperature-dependent, and the
   2026 review found that no system it appraised demonstrated prospectively
   validated environmental adaptation of thresholds. Temperature is available
   here for free, in every filename.

Labelling
---------
TreeVibes assigns labels *by folder*, not by filename. The published mapping
(Rigakis et al. 2021, Section 5) is:

    infested : folders 1-6 and 11-23
    clean    : folders 7-10, 24-25, 35

The Kaggle distribution may instead use folder names containing the words, or
ship an annotation CSV. `load()` tries each scheme in turn and reports which
one it used, so the provenance of every label is explicit.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".opus", ".m4a"}

# Published folder -> label mapping from the TreeVibes paper.
PAPER_INFESTED_FOLDERS = set(range(1, 7)) | set(range(11, 24))
PAPER_CLEAN_FOLDERS = {7, 8, 9, 10, 24, 25, 35}

CLEAN_WORDS = [r"\bclean\b", r"\bhealthy\b", r"\bnot[_\- ]?infest",
               r"\bnon[_\- ]?infest", r"\bno[_\- ]?infest", r"\bnegative\b",
               r"\bnormal\b", r"\bcontrol\b", r"\bfree\b"]
INFESTED_WORDS = [r"\binfest", r"\bpest\b", r"\bweevil\b", r"\brpw\b",
                  r"\bpositive\b", r"\bborer\b", r"\blarva", r"\bxylo"]

# F_20200222172127_4_T14.0  ->  timestamp, index, temperature
FNAME_RE = re.compile(
    r"(?P<prefix>[A-Za-z]+)?_?"
    r"(?P<ts>\d{14})"
    r"(?:_(?P<idx>\d+))?"
    r"(?:_T(?P<temp>-?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# filename metadata
# --------------------------------------------------------------------------

def parse_filename(name: str) -> dict:
    """Extract timestamp, index and temperature from a TreeVibes filename.

    Returns NaT/NaN for any field the name does not carry, so this is safe to
    apply to files from other datasets.
    """
    stem = Path(name).stem
    m = FNAME_RE.search(stem)
    if not m:
        return {"timestamp": pd.NaT, "file_index": np.nan, "temperature_c": np.nan}

    ts = pd.to_datetime(m.group("ts"), format="%Y%m%d%H%M%S", errors="coerce")
    idx = m.group("idx")
    temp = m.group("temp")
    return {
        "timestamp": ts,
        "file_index": float(idx) if idx is not None else np.nan,
        "temperature_c": float(temp) if temp is not None else np.nan,
    }


# --------------------------------------------------------------------------
# structural inspection
# --------------------------------------------------------------------------

def inspect(root, max_depth: int = 4, examples: int = 3) -> str:
    """Report what is actually inside a download, before trying to load it.

    Run this first on any new dataset. It prints the directory tree, audio
    counts, any CSVs, and sample filenames -- enough to see which labelling
    scheme applies without guessing.
    """
    root = Path(root).resolve()
    if not root.exists():
        return f"Path does not exist: {root}"

    lines = [f"TreeVibes inspection: {root}", "=" * 72]

    audio, csvs, others = [], [], Counter()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        if suf in AUDIO_EXT:
            audio.append(p)
        elif suf in {".csv", ".tsv", ".xlsx", ".xls"}:
            csvs.append(p)
        else:
            others[suf or "(no extension)"] += 1

    lines.append(f"audio files : {len(audio)}")
    lines.append(f"tables      : {len(csvs)}")
    if others:
        lines.append(f"other       : {dict(others.most_common(6))}")

    if not audio:
        lines.append("\nNo audio found. Check that the zip was extracted here.")
        top = sorted({p.relative_to(root).parts[0]
                      for p in root.rglob("*") if p.is_file()})[:25]
        lines.append(f"top-level entries: {top}")
        return "\n".join(lines)

    # ---- directory shape -------------------------------------------------
    lines.append("\nDirectory structure (audio-bearing folders)")
    lines.append("-" * 72)
    by_dir = Counter(p.parent.relative_to(root).as_posix() or "." for p in audio)
    shown = 0
    for d, n in sorted(by_dir.items()):
        depth = 0 if d == "." else d.count("/") + 1
        if depth <= max_depth:
            lines.append(f"  {d or '.':<52} {n:>6} files")
            shown += 1
        if shown >= 40:
            lines.append(f"  ... and {len(by_dir) - shown} more folders")
            break
    lines.append(f"  total folders containing audio: {len(by_dir)}")

    # ---- filename metadata ----------------------------------------------
    lines.append("\nFilename examples")
    lines.append("-" * 72)
    for p in audio[:examples]:
        meta = parse_filename(p.name)
        lines.append(f"  {p.name}")
        lines.append(f"      timestamp={meta['timestamp']}  "
                     f"temp={meta['temperature_c']}")

    parsed = [parse_filename(p.name) for p in audio[:4000]]
    n_ts = sum(1 for m in parsed if pd.notna(m["timestamp"]))
    n_temp = sum(1 for m in parsed if pd.notna(m["temperature_c"]))
    lines.append(f"  parsed timestamps  : {n_ts}/{len(parsed)} sampled")
    lines.append(f"  parsed temperatures: {n_temp}/{len(parsed)} sampled")
    if n_ts:
        ts = pd.Series([m["timestamp"] for m in parsed]).dropna()
        lines.append(f"  date range (sample): {ts.min()}  ->  {ts.max()}")

    # ---- label hints -----------------------------------------------------
    lines.append("\nLabelling hints")
    lines.append("-" * 72)
    rels = [p.relative_to(root).as_posix().lower() for p in audio]
    n_clean = sum(any(re.search(w, r) for w in CLEAN_WORDS) for r in rels[:4000])
    n_inf = sum(any(re.search(w, r) for w in INFESTED_WORDS) for r in rels[:4000])
    lines.append(f"  paths matching clean words   : {n_clean}")
    lines.append(f"  paths matching infested words: {n_inf}")
    folder_nums = {_folder_number(p, root) for p in audio}
    folder_nums.discard(None)
    if folder_nums:
        lines.append(f"  numeric folder ids found     : "
                     f"{sorted(folder_nums)[:20]}"
                     f"{' ...' if len(folder_nums) > 20 else ''}")

    if csvs:
        lines.append("\nAnnotation tables")
        lines.append("-" * 72)
        for c in csvs[:5]:
            lines.append(f"  {c.relative_to(root)}")
            try:
                head = (pd.read_excel(c, nrows=3)
                        if c.suffix.lower() in {".xlsx", ".xls"}
                        else pd.read_csv(c, nrows=3))
                lines.append(f"      columns: {list(head.columns)[:12]}")
                lines.append(f"      first row: "
                             f"{head.iloc[0].to_dict() if len(head) else '(empty)'}")
            except Exception as exc:
                lines.append(f"      (could not read: {type(exc).__name__})")

    lines.append("\n" + "=" * 72)
    lines.append("Next: treevibes.load(root) will use these to assign labels.")
    return "\n".join(lines)


def _folder_number(path: Path, root: Path):
    """First integer-like directory component under root, if any."""
    for part in path.relative_to(root).parts[:-1]:
        m = re.fullmatch(r"(?:folder|tree|f|t)?[_\- ]?(\d{1,3})", part.strip(),
                         re.IGNORECASE)
        if m:
            return int(m.group(1))
    return None


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _label_from_words(rel: str):
    r = rel.lower().replace("-", "_")
    if any(re.search(w, r) for w in CLEAN_WORDS):
        return "clean"
    if any(re.search(w, r) for w in INFESTED_WORDS):
        return "infested"
    return None


def _group_from_path(path: Path, root: Path) -> str:
    """The recording folder: the tree/session a clip belongs to.

    Label directories are excluded so that a layout like <label>/<folder>/
    still groups by folder rather than by class.
    """
    parts = list(path.relative_to(root).parts[:-1])
    keep = [p for p in parts if _label_from_words(p) is None]
    if keep:
        return "/".join(keep)
    return parts[0] if parts else path.stem


def load(root, csv_path=None, folder_labels: dict | None = None,
         use_paper_mapping: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Build a labelled manifest of TreeVibes recordings.

    Label sources are tried in this order, and the one used is recorded in
    the `label_source` column so every label's provenance is explicit:

      1. `folder_labels`  - a mapping you supply, e.g. {1: "infested", 7: "clean"}
      2. `csv_path`       - an annotation table shipped with the dataset
      3. path keywords    - "clean"/"infested" anywhere in the path
      4. paper mapping    - the folder numbers published in Rigakis et al. 2021

    Returns a DataFrame with: path, label, y, group, timestamp,
    temperature_c, file_index, label_source.
    """
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root}\n"
            "Download TreeVibes from https://www.kaggle.com/datasets/potamitis/treevibes "
            "and extract it there."
        )

    audio = [p for p in sorted(root.rglob("*"))
             if p.is_file() and p.suffix.lower() in AUDIO_EXT]
    if not audio:
        raise ValueError(
            f"No audio files under {root}.\n"
            "Run treevibes.inspect(root) to see what is actually there."
        )

    # ---- optional CSV ----------------------------------------------------
    csv_lookup = {}
    if csv_path is None:
        # The Kaggle distribution ships TreeVibes.xlsx, not a CSV. Look for
        # both; an .xlsx that is silently skipped sends the loader off to
        # guess labels from the path instead of using the authors' own
        # annotations, which is a quiet and serious error.
        #
        # Search upward as well as down. The annotations sit at the dataset
        # root (data/treevibes/TreeVibes.xlsx) while analysis runs against a
        # subfolder (data/treevibes/field/), so a downward-only search misses
        # the table entirely -- which is exactly what happened on the first
        # real run, discarding every test folder for want of a label.
        def _tables(d: Path):
            return ([q for q in d.glob("*.xlsx") if not q.name.startswith("~$")]
                    + [q for q in d.glob("*.csv")])

        found = ([q for q in root.rglob("*.xlsx") if not q.name.startswith("~$")]
                 + [q for q in root.rglob("*.csv")])
        for parent in list(root.parents)[:2]:        # treevibes/, data/
            found += _tables(parent)
        csv_path = found[0] if found else None
    if csv_path is not None:
        try:
            csv_lookup = _labels_from_csv(Path(csv_path))
            if verbose and csv_lookup:
                print(f"  annotation table: {Path(csv_path).name} "
                      f"({len(csv_lookup)} folder labels)")
        except Exception as exc:
            if verbose:
                print(f"  note: could not use {Path(csv_path).name}: "
                      f"{type(exc).__name__}: {exc}")

    rows, unlabelled = [], []
    for p in audio:
        rel = p.relative_to(root).as_posix()
        fnum = _folder_number(p, root)
        label = source = None

        if folder_labels and fnum in folder_labels:
            label, source = folder_labels[fnum], "user_mapping"
        if label is None and csv_lookup and fnum in csv_lookup:
            label, source = csv_lookup[fnum], "annotation_csv"
        if label is None:
            label = _label_from_words(rel)
            source = "path_keyword" if label else None
        if label is None and use_paper_mapping and fnum is not None:
            if fnum in PAPER_INFESTED_FOLDERS:
                label, source = "infested", "paper_mapping"
            elif fnum in PAPER_CLEAN_FOLDERS:
                label, source = "clean", "paper_mapping"

        if label is None:
            unlabelled.append(rel)
            continue

        meta = parse_filename(p.name)
        rows.append({
            "path": str(p),
            "label": label,
            "y": 1 if label == "infested" else 0,
            "group": _group_from_path(p, root),
            "folder_number": fnum,
            "label_source": source,
            **meta,
        })

    if not rows:
        raise ValueError(
            "No file could be labelled.\n"
            f"  {len(audio)} audio files found, none matched any scheme.\n"
            "  Run treevibes.inspect(root) and pass folder_labels={...} "
            "explicitly, e.g. {1: 'infested', 7: 'clean'}."
        )

    df = pd.DataFrame(rows).sort_values(["group", "timestamp"]).reset_index(drop=True)

    if verbose:
        print(f"  labelled {len(df)} of {len(audio)} audio files")
        if unlabelled:
            print(f"  {len(unlabelled)} unlabelled (e.g. {unlabelled[0]})")
        print(f"  label sources: {df.label_source.value_counts().to_dict()}")

    # Where the dataset states the same fact twice -- once in the annotation
    # table, once in its train/clean and train/infested folders -- the two
    # must agree. This check exists because they once did not, and nothing
    # said so.
    if (df["label_source"] == "annotation_csv").any():
        _cross_check(df, verbose=verbose)
    return df


def _labels_from_csv(csv_path: Path, verbose: bool = True) -> dict:
    """Map folder number -> label from TreeVibes' annotation table.

    The real table has columns:

        FOLDER | IMEI | GPS_LAT | GPS_LONG | VISUAL | AUDIO | CONFIRMATION | COMMENT

    **AUDIO is the label**: 0 = clean, 1 = infested. It agrees exactly with
    the dataset's own train/clean and train/infested directory split, which
    is how it was verified.

    The other two columns look like labels and are not:

      CONFIRMATION is YES/NO for whether the diagnosis was *verified*, not
      whether the tree is infested. folder_7 is a healthy orange tree with
      CONFIRMATION=YES. An earlier version of this function used this column
      and produced clean=68 / infested=4871 -- an almost perfect inversion of
      the truth, silently, with no error raised.

      VISUAL is a three-way external-inspection grade (A heavy, B low, X no
      visible signs, in a mix of Greek and Latin letters). It disagrees with
      AUDIO on several folders, because a tree can be acoustically active
      with no exit tunnels yet.

    So the search is deliberately narrow: a column actually named AUDIO whose
    values are binary. Guessing from column-name fragments is what caused the
    inversion, and breadth here is a liability rather than a convenience.
    """
    if csv_path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(csv_path)
        except ImportError as exc:
            raise ImportError(
                f"Reading {csv_path.name} needs openpyxl:\n"
                "    pip install openpyxl"
            ) from exc
    else:
        df = pd.read_csv(csv_path)

    cols = {str(c).strip().lower(): c for c in df.columns}

    folder_col = cols.get("folder")
    if folder_col is None:
        for lc, orig in cols.items():
            if "folder" in lc:
                folder_col = orig
                break
    if folder_col is None:
        return {}

    # The label column must be named AUDIO *and* be binary. Both conditions,
    # not either.
    label_col = None
    for lc, orig in cols.items():
        if lc != "audio" and "audio" not in lc:
            continue
        vals = set(pd.to_numeric(df[orig], errors="coerce").dropna().unique())
        if vals and vals <= {0.0, 1.0}:
            label_col = orig
            break
    if label_col is None:
        if verbose:
            print("  note: no binary AUDIO column found in the annotation "
                  "table; falling back to path keywords")
        return {}

    out = {}
    for _, r in df.iterrows():
        m = re.search(r"(\d{1,3})", str(r[folder_col]))
        if not m:
            continue
        v = pd.to_numeric(r[label_col], errors="coerce")
        if pd.isna(v):
            continue
        out[int(m.group(1))] = "infested" if int(v) == 1 else "clean"
    return out


def _cross_check(df: pd.DataFrame, verbose: bool = True) -> dict:
    """Compare table labels against the dataset's own directory structure.

    TreeVibes puts its training folders under train/clean and train/infested,
    which is an independent statement of the same fact the annotation table
    records. Where both exist they must agree. A disagreement means the table
    is being read wrongly, and that is worth stopping for -- an inverted label
    set produces confident, completely meaningless results.
    """
    checked = agree = 0
    disagreements = []
    for _, r in df.iterrows():
        path_label = _label_from_words(Path(r["path"]).as_posix())
        if path_label is None:
            continue
        checked += 1
        if path_label == r["label"]:
            agree += 1
        elif len(disagreements) < 5:
            disagreements.append((r["group"], path_label, r["label"]))

    result = {"checked": checked, "agree": agree,
              "rate": agree / checked if checked else float("nan"),
              "examples": disagreements}

    if verbose and checked:
        rate = result["rate"]
        print(f"  cross-check vs folder structure: {agree}/{checked} agree "
              f"({rate:.1%})")
        if rate < 0.95:
            print("  WARNING: the annotation table disagrees with the "
                  "dataset's own train/clean and train/infested folders.")
            print("  The table is probably being read from the wrong column. "
                  "Do not use these labels.")
            for g, pl, tl in disagreements:
                print(f"    {g}: folders say {pl}, table says {tl}")
    return result


# --------------------------------------------------------------------------
# description
# --------------------------------------------------------------------------

def describe(df: pd.DataFrame) -> str:
    counts = df["label"].value_counts().to_dict()
    per_group = df.groupby("group")["label"].nunique()
    mixed = int((per_group > 1).sum())

    lines = [
        f"  recordings        : {len(df)}",
        "  class balance     : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"  recording groups  : {df['group'].nunique()}",
        f"  groups with both classes : {mixed}",
    ]
    if "timestamp" in df and df["timestamp"].notna().any():
        ts = df["timestamp"].dropna()
        span = (ts.max() - ts.min()).days
        lines.append(f"  time span         : {ts.min():%Y-%m-%d} to "
                     f"{ts.max():%Y-%m-%d}  ({span} days)")
        per_tree_days = df.groupby("group")["timestamp"].agg(
            lambda s: (s.max() - s.min()).days if s.notna().any() else 0)
        lines.append(f"  days per tree     : median {per_tree_days.median():.0f}, "
                     f"max {per_tree_days.max():.0f}")
    if "temperature_c" in df and df["temperature_c"].notna().any():
        t = df["temperature_c"].dropna()
        lines.append(f"  temperature       : {t.min():.1f} to {t.max():.1f} C "
                     f"(median {t.median():.1f})")
    if mixed == 0:
        lines.append("")
        lines.append("  NOTE: no tree is recorded in both states, so the")
        lines.append("  within-tree contrast is unavailable. The per-tree")
        lines.append("  method falls back to a pooled estimate; say so.")
    return "\n".join(lines)


def daily_sequences(df: pd.DataFrame, min_obs: int = 20) -> pd.DataFrame:
    """Collapse a manifest to one row per tree-day, preserving real time.

    A device recorded several times a day. For longitudinal analysis the
    natural unit is the day, which also evens out diurnal activity cycles.
    """
    if "timestamp" not in df or df["timestamp"].isna().all():
        raise ValueError("No timestamps available; cannot build daily sequences.")
    d = df.dropna(subset=["timestamp"]).copy()
    d["date"] = d["timestamp"].dt.floor("D")
    keep = d.groupby("group")["date"].transform("count") >= min_obs
    return d[keep].sort_values(["group", "timestamp"]).reset_index(drop=True)
