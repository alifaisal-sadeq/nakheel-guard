"""Train and honestly evaluate the Nakheel Guard classifier.

Usage
-----
    python src/train.py --data data/treevibes
    python src/train.py --data data/treevibes --tag treevibes --refresh

The script runs the same models under two evaluation protocols:

  grouped  - test folds contain only recording groups (trees) absent from
             training. This is the deployment-realistic number.
  random   - clips shuffled without regard to their recording. This is the
             protocol that produces the near-perfect figures reported in the
             literature, included so the gap can be measured.

Everything written to results/ is derived from the grouped protocol unless
explicitly labelled otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import RESULTS_DIR, RANDOM_SEED, N_SPLITS, FALSE_ALARM_BUDGET, MODEL_PATH
import data as data_mod
from extract import build_matrix, save_cache, load_cache


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

def make_models(with_gbm: bool = False):
    """Models of increasing capacity, all class-balanced.

    Class weighting matters here: TreeVibes is heavily skewed, and an
    unweighted model can score well on accuracy while never predicting the
    minority class at all.

    Gradient boosting is off by default. On a laptop it costs roughly 25x
    more time per fit than the random forest for little or no gain on this
    feature set; enable it with --with-gbm once the rest of the run is
    settled and you have time to spare.
    """
    models = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=None)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_SEED, n_jobs=-1),
    }
    if with_gbm:
        models["gradient_boosting"] = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15,
            random_state=RANDOM_SEED, class_weight="balanced")
    return models


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def threshold_for_false_alarm_budget(y_true, scores, budget=FALSE_ALARM_BUDGET):
    """Lowest threshold whose false-positive rate stays within budget.

    Chosen on the training fold only. Missing an infested tree is far more
    costly than inspecting a healthy one, so we accept false alarms up to a
    stated budget and maximise recall underneath it.
    """
    clean_scores = scores[y_true == 0]
    if clean_scores.size == 0:
        return 0.5
    return float(np.quantile(clean_scores, 1.0 - budget))


def score_fold(y_true, y_pred, scores) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, scores)
        out["pr_auc"] = average_precision_score(y_true, scores)
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def _scores_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def cross_validate(X, y, groups, protocol: str, n_splits=N_SPLITS,
                   with_gbm: bool = False) -> dict:
    """Run every model under one evaluation protocol."""
    df = pd.DataFrame({"y": y, "group": groups})
    splits = list(
        data_mod.grouped_splits(df, n_splits) if protocol == "grouped"
        else data_mod.random_splits(df, n_splits)
    )

    results = {}
    for name in make_models(with_gbm):
        fold_rows = []
        for k, (tr, te) in enumerate(splits):
            if len(np.unique(y[tr])) < 2:
                continue
            model = make_models(with_gbm)[name]
            model.fit(X[tr], y[tr])

            train_scores = _scores_of(model, X[tr])
            thr = threshold_for_false_alarm_budget(y[tr], train_scores)

            test_scores = _scores_of(model, X[te])
            y_pred = (test_scores >= thr).astype(int)
            row = score_fold(y[te], y_pred, test_scores)
            row["fold"] = k
            row["threshold"] = thr
            row["n_test"] = int(len(te))
            row["n_test_groups"] = int(pd.Series(groups[te]).nunique())
            fold_rows.append(row)

        if not fold_rows:
            continue
        folds = pd.DataFrame(fold_rows)
        numeric = folds.select_dtypes(include=[np.number])
        results[name] = {
            "folds": folds.to_dict(orient="records"),
            "mean": numeric.mean().to_dict(),
            "std": numeric.std(ddof=0).to_dict(),
        }
    return results


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

KEY_METRICS = ["balanced_accuracy", "recall", "specificity", "precision",
               "f1", "roc_auc", "pr_auc"]


def print_table(title: str, results: dict):
    print(f"\n{title}")
    print("-" * 78)
    header = f"{'model':<22}" + "".join(f"{m[:9]:>11}" for m in KEY_METRICS)
    print(header)
    for name, res in results.items():
        m, s = res["mean"], res["std"]
        line = f"{name:<22}"
        for metric in KEY_METRICS:
            val, err = m.get(metric, float('nan')), s.get(metric, 0.0)
            line += f"{val:>7.3f}±{err:<3.2f}"
        print(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="path to the dataset root")
    ap.add_argument("--tag", default="dataset", help="name for cached features")
    ap.add_argument("--refresh", action="store_true", help="ignore cached features")
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--with-gbm", action="store_true",
                    help="also evaluate gradient boosting (much slower)")
    ap.add_argument("--limit-per-class", type=int, default=None,
                    help="cap clips per class (sampled across all recording "
                         "groups). Start with 1500 on TreeVibes to get a "
                         "result in minutes, then raise it.")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("NAKHEEL GUARD  |  acoustic detection of Red Palm Weevil")
    print("=" * 78)

    print("\n[1/4] Scanning dataset")
    df = data_mod.scan_dataset(args.data)
    print(data_mod.describe(df))
    if args.limit_per_class:
        df = data_mod.subsample(df, args.limit_per_class)
        print(f"\n  subsampled to at most {args.limit_per_class} clips per class:")
        print(data_mod.describe(df))

    print("\n[2/4] Extracting features")
    cached = None if args.refresh else load_cache(args.tag)
    if cached is not None:
        X, y, groups, clip_ids, names = cached
        print(f"  loaded cache: {X.shape[0]} windows x {X.shape[1]} features")
    else:
        X, y, groups, clip_ids, names = build_matrix(df, workers=args.workers)
        save_cache(args.tag, X, y, groups, clip_ids, names)
        print(f"  extracted   : {X.shape[0]} windows x {X.shape[1]} features")
    print(f"  windows per class: clean={int((y==0).sum())}, infested={int((y==1).sum())}")

    print("\n[3/4] Cross-validation")
    grouped = cross_validate(X, y, groups, "grouped", args.splits, args.with_gbm)
    random_ = cross_validate(X, y, groups, "random", args.splits, args.with_gbm)

    print_table("GROUPED SPLIT  (test trees unseen in training) -- the honest number",
                grouped)
    print_table("RANDOM SPLIT   (clips shuffled; leakage-prone) -- literature protocol",
                random_)

    best = max(grouped, key=lambda n: grouped[n]["mean"]["balanced_accuracy"])
    gap = (random_[best]["mean"]["balanced_accuracy"]
           - grouped[best]["mean"]["balanced_accuracy"])
    print("\n" + "-" * 78)
    print(f"Best model (grouped): {best}")
    print(f"  balanced accuracy  grouped {grouped[best]['mean']['balanced_accuracy']:.3f}"
          f"   random {random_[best]['mean']['balanced_accuracy']:.3f}"
          f"   gap {gap:+.3f}")
    print("  The gap is the share of apparent performance that comes from")
    print("  recognising the recording rather than the insect.")

    print("\n[4/4] Fitting final model on all data")
    final = make_models(args.with_gbm)[best]
    final.fit(X, y)
    train_scores = _scores_of(final, X)
    thr = threshold_for_false_alarm_budget(y, train_scores)
    import joblib
    joblib.dump({"model": final, "threshold": thr, "feature_names": names,
                 "model_name": best}, MODEL_PATH)
    print(f"  saved {MODEL_PATH}")

    report = {
        "dataset": str(Path(args.data).resolve()),
        "n_clips": int(len(df)),
        "n_windows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_groups": int(df["group"].nunique()),
        "class_counts": {k: int(v) for k, v in df["label"].value_counts().items()},
        "best_model": best,
        "decision_threshold": thr,
        "false_alarm_budget": FALSE_ALARM_BUDGET,
        "grouped": grouped,
        "random": random_,
        "leakage_gap_balanced_accuracy": gap,
    }
    out = RESULTS_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2, default=float))
    print(f"  saved {out}")
    print("\nDone. Use results/report.json for the poster numbers.")


if __name__ == "__main__":
    main()
