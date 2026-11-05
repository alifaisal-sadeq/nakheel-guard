"""Generate the figures that go on the scientific poster.

    python src/plots.py            # uses results/report.json

Produces, in results/:
  fig_leakage.png     grouped vs random evaluation, the headline result
  fig_confusion.png   confusion matrix of the best model, grouped protocol
  fig_features.png    which feature families carry the signal
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS_DIR

TEAL = "#123F3B"
MID = "#2E6B63"
CREAM = "#F3F0E1"
ALERT = "#C0392B"
AMBER = "#E08A3C"
GREY = "#8FA3A0"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#C8D6D0",
    "axes.labelcolor": TEAL,
    "text.color": TEAL,
    "xtick.color": TEAL,
    "ytick.color": TEAL,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

METRICS = [("balanced_accuracy", "Balanced\naccuracy"),
           ("recall", "Recall\n(infested caught)"),
           ("specificity", "Specificity\n(clean correct)"),
           ("roc_auc", "ROC-AUC")]


def load_report():
    p = RESULTS_DIR / "report.json"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run src/train.py first.")
    return json.loads(p.read_text())


def fig_leakage(rep):
    """The headline figure: what grouped evaluation costs you."""
    best = rep["best_model"]
    g, r = rep["grouped"][best], rep["random"][best]

    labels = [lab for _, lab in METRICS]
    gv = [g["mean"].get(k, np.nan) for k, _ in METRICS]
    ge = [g["std"].get(k, 0.0) for k, _ in METRICS]
    rv = [r["mean"].get(k, np.nan) for k, _ in METRICS]
    re_ = [r["std"].get(k, 0.0) for k, _ in METRICS]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.bar(x - w / 2, rv, w, yerr=re_, capsize=4, color=GREY,
           label="Random split  (clips shuffled — literature protocol)",
           error_kw={"ecolor": "#5E7370", "lw": 1.2})
    ax.bar(x + w / 2, gv, w, yerr=ge, capsize=4, color=MID,
           label="Grouped split  (test trees unseen in training)",
           error_kw={"ecolor": TEAL, "lw": 1.2})

    ax.axhline(0.5, color=ALERT, ls=":", lw=1.4, zorder=0)
    ax.text(-0.46, 0.52, "chance", color=ALERT, fontsize=9)

    # Place each label above the top of its error bar, not its bar, so the
    # two never overlap.
    for xi in range(len(labels)):
        if np.isfinite(rv[xi]):
            ax.text(xi - w / 2, rv[xi] + re_[xi] + 0.03, f"{rv[xi]:.2f}",
                    ha="center", fontsize=9.5, color="#5E7370")
        if np.isfinite(gv[xi]):
            ax.text(xi + w / 2, gv[xi] + ge[xi] + 0.03, f"{gv[xi]:.2f}",
                    ha="center", fontsize=9.5, fontweight="bold", color=TEAL)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("score")
    ax.set_title(
        "The same model, two evaluation protocols",
        fontsize=14, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper center", fontsize=9.5,
              bbox_to_anchor=(0.5, -0.11), ncol=1)
    fig.tight_layout()
    out = RESULTS_DIR / "fig_leakage.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_confusion(rep):
    best = rep["best_model"]
    folds = rep["grouped"][best]["folds"]
    tn = sum(f["tn"] for f in folds); fp = sum(f["fp"] for f in folds)
    fn = sum(f["fn"] for f in folds); tp = sum(f["tp"] for f in folds)
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(norm, cmap="BuGn", vmin=0, vmax=1)
    names = ["clean", "infested"]
    ax.set_xticks([0, 1], labels=[f"predicted\n{n}" for n in names])
    ax.set_yticks([0, 1], labels=[f"actual\n{n}" for n in names])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j])}\n{norm[i, j]*100:.0f}%",
                    ha="center", va="center", fontsize=14,
                    color="white" if norm[i, j] > 0.55 else TEAL,
                    fontweight="bold")
    ax.set_title("Grouped evaluation\n(pooled across folds)",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "fig_confusion.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_features():
    """Permutation importance grouped by feature family.

    Answers a question judges ask: which part of the signal matters? In
    particular, do the hand-designed burst statistics earn their place
    against generic audio features?
    """
    cache = sorted(RESULTS_DIR.glob("features_*.npz"))
    model_p = RESULTS_DIR / "model.joblib"
    if not cache or not model_p.exists():
        return None

    import joblib
    from sklearn.inspection import permutation_importance

    d = np.load(cache[-1], allow_pickle=True)
    X, y, names = d["X"], d["y"], list(d["names"])
    bundle = joblib.load(model_p)
    model = bundle["model"]

    # Permutation importance on a capped subsample keeps this quick.
    rng = np.random.RandomState(0)
    n = min(1200, X.shape[0])
    idx = rng.choice(X.shape[0], n, replace=False)
    imp = permutation_importance(model, X[idx], y[idx], n_repeats=5,
                                 random_state=0, scoring="balanced_accuracy")

    def family(name: str) -> str:
        if name.startswith("burst_"):
            return "Impulse statistics\n(domain-designed)"
        if name.startswith("mel_"):
            return "Log-mel bands"
        if name.startswith("dmfcc_"):
            return "MFCC deltas"
        if name.startswith("mfcc_"):
            return "MFCC"
        return "Spectral shape"

    totals: dict[str, float] = {}
    for name, val in zip(names, imp.importances_mean):
        totals[family(name)] = totals.get(family(name), 0.0) + max(val, 0.0)

    order = sorted(totals, key=totals.get)
    vals = [totals[k] for k in order]
    colours = [ALERT if "domain-designed" in k else MID for k in order]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(order, vals, color=colours, height=0.6)
    ax.set_xlabel("summed permutation importance (balanced accuracy)")
    ax.set_title("Where the signal comes from", fontsize=13,
                 fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    out = RESULTS_DIR / "fig_features.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_detection_delay(df, out_name="fig_detection_delay.png"):
    """Poster figure for the sequential-detection result.

    Left: detection rate against infestation prevalence.
    Right: median delay from onset to alarm. Lower is better.
    Both detectors are held to the same false-alarm budget, stated on the
    figure, because otherwise the comparison would mean nothing.
    """
    a = df.attrs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    x = df["final_prevalence"].values * 100

    ax1.plot(x, df["cusum_detection_rate"], "o-", color=MID, lw=2.2, ms=6,
             label="Sequential (this work)")
    ax1.plot(x, df["perclip_detection_rate"], "s--", color=GREY, lw=2.0, ms=5,
             label="Per-clip classification")
    ax1.set_ylabel("detection rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Infestations detected", fontsize=12,
                  fontweight="bold", loc="left")

    ax2.plot(x, df["cusum_median_delay"], "o-", color=MID, lw=2.2, ms=6,
             label="Sequential (this work)")
    ax2.plot(x, df["perclip_median_delay"], "s--", color=GREY, lw=2.0, ms=5,
             label="Per-clip classification")
    ax2.set_ylabel("median delay to alarm (observations)")
    ax2.set_title("How long until the alarm", fontsize=12,
                  fontweight="bold", loc="left")

    for ax in (ax1, ax2):
        ax.set_xlabel("infestation prevalence (% of recordings with feeding)")
        ax.grid(axis="y", color="#E3EAE7", lw=0.8)
        ax.set_axisbelow(True)

    ax1.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.suptitle(
        f"Both detectors held to the same budget: "
        f"{a['false_alarm_budget']:.2f} false alarms per tree per "
        f"{a['n_obs']} observations",
        fontsize=10, color="#5E7370", y=0.015)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_stream_example(scores, p_curve, S, h, onset, alarm,
                       out_name="fig_stream_example.png"):
    """Illustrate one monitoring stream: why accumulation beats thresholding."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True,
                                   height_ratios=[1, 1])
    t = np.arange(len(scores))

    ax1.plot(t, scores, lw=0.9, color=GREY, label="per-observation score")
    ax1.axvline(onset, color=AMBER, lw=1.6, ls="-")
    ax1.text(onset + 1, ax1.get_ylim()[1] * 0.88, " infestation begins",
             color=AMBER, fontsize=9)
    ax1.set_ylabel("score")
    ax1.set_title("A single observation is mostly noise",
                  fontsize=12, fontweight="bold", loc="left")

    ax1b = ax1.twinx()
    ax1b.plot(t, p_curve, color=ALERT, lw=1.6, ls=":")
    ax1b.set_ylabel("true prevalence", color=ALERT, fontsize=9)
    ax1b.tick_params(axis="y", colors=ALERT, labelsize=8)
    ax1b.set_ylim(0, 1.05)

    ax2.plot(t, S, color=MID, lw=2.0)
    ax2.axhline(h, color=ALERT, ls="--", lw=1.4)
    ax2.text(1, h * 1.04, "alarm threshold", color=ALERT, fontsize=9)
    ax2.axvline(onset, color=AMBER, lw=1.6)
    if alarm is not None:
        ax2.axvline(alarm, color=TEAL, lw=1.6, ls="-.")
        ax2.text(alarm + 1, h * 0.45,
                 f" detected\n +{alarm - onset} obs", color=TEAL, fontsize=9)
    ax2.set_ylabel("CUSUM statistic")
    ax2.set_xlabel("observation (one per day)")
    ax2.set_title("Accumulated evidence crosses it",
                  fontsize=12, fontweight="bold", loc="left")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#E3EAE7", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    rep = load_report()
    made = [fig_leakage(rep), fig_confusion(rep)]
    f = fig_features()
    if f:
        made.append(f)
    for p in made:
        print(f"  wrote {p}")
    print("\nfig_leakage.png is the one for the poster's Results section.")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------
# temperature / seasonal figures
# --------------------------------------------------------------------------

def fig_temperature_response(baselines: dict, out_name="fig_temperature_response.png"):
    """Fitted activity-vs-temperature curves, one line per tree.

    If these slope upward, acoustic activity is temperature-dependent and a
    detector that ignores temperature will misread seasonal warming as
    infestation.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, (name, bl) in enumerate(baselines.items()):
        t, c = bl.curve()
        ax.plot(t, c, lw=1.8, alpha=0.85,
                color=MID if i % 2 == 0 else TEAL,
                label=name if len(baselines) <= 8 else None)
    ax.set_xlabel("trunk temperature (°C)")
    ax.set_ylabel("expected healthy score")
    ax.set_title("Acoustic activity depends on temperature",
                 fontsize=13, fontweight="bold", loc="left", pad=12)
    ax.grid(axis="y", color="#E3EAE7", lw=0.8)
    ax.set_axisbelow(True)
    if len(baselines) <= 8:
        ax.legend(frameon=False, fontsize=8.5, ncol=2)
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_confound(conf: pd.DataFrame, out_name="fig_temperature_confound.png"):
    """Seasonal drift of a temperature-blind detector, on healthy trees.

    The y axis is the mean residual during monitoring. Zero is correct. A
    temperature-blind detector drifts upward as the season warms; the
    temperature-aware one should not.
    """
    healthy = conf[conf["label"] == "clean"]
    if healthy.empty:
        healthy = conf

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    x = np.arange(len(healthy))
    wdt = 0.38
    ax1.bar(x - wdt / 2, healthy["naive_residual_mean"], wdt,
            color=ALERT, label="temperature-blind")
    ax1.bar(x + wdt / 2, healthy["aware_residual_mean"], wdt,
            color=MID, label="temperature-aware (this work)")
    ax1.axhline(0, color=TEAL, lw=1.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(healthy["group"], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("mean residual during monitoring")
    ax1.set_title("Healthy trees — drift should be zero", fontsize=12,
                  fontweight="bold", loc="left")
    ax1.legend(frameon=False, fontsize=9)

    ax2.scatter(conf["temp_shift"], conf["naive_residual_mean"],
                s=46, color=ALERT, label="temperature-blind", zorder=3)
    ax2.scatter(conf["temp_shift"], conf["aware_residual_mean"],
                s=46, color=MID, marker="s",
                label="temperature-aware", zorder=3)
    ax2.axhline(0, color=TEAL, lw=1.2)
    ax2.set_xlabel("warming between learning and monitoring (°C)")
    ax2.set_ylabel("mean residual")
    ax2.set_title("Drift tracks the warming", fontsize=12,
                  fontweight="bold", loc="left")
    ax2.legend(frameon=False, fontsize=9)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color="#E3EAE7", lw=0.8)
        ax.set_axisbelow(True)
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_real_timeline(stream, baseline, split_date, label,
                      out_name="fig_real_timeline.png"):
    """One tree's real monitoring record: score, temperature, residuals."""
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.2), sharex=True)
    d = stream["date"]

    axes[0].plot(d, stream["score"], lw=1.0, color=GREY)
    axes[0].set_ylabel("daily score")
    axes[0].set_title(f"{label} — raw score", fontsize=11,
                      fontweight="bold", loc="left")

    axes[1].plot(d, stream["temperature_c"], lw=1.4, color=AMBER)
    axes[1].set_ylabel("°C")
    axes[1].set_title("Temperature", fontsize=11, fontweight="bold", loc="left")

    naive = baseline.naive_residual(stream["score"].values)
    aware = baseline.residual(stream["score"].values,
                              stream["temperature_c"].values)
    axes[2].plot(d, naive, lw=1.2, color=ALERT, label="temperature-blind")
    axes[2].plot(d, aware, lw=1.4, color=MID, label="temperature-aware")
    axes[2].axhline(0, color=TEAL, lw=1.0)
    axes[2].set_ylabel("residual")
    axes[2].set_title("Residual — zero means 'as expected'", fontsize=11,
                      fontweight="bold", loc="left")
    axes[2].legend(frameon=False, fontsize=9)

    for ax in axes:
        ax.axvline(split_date, color=TEAL, ls="--", lw=1.4)
        ax.grid(axis="y", color="#E3EAE7", lw=0.8)
        ax.set_axisbelow(True)
    axes[0].text(split_date, axes[0].get_ylim()[1] * 0.92,
                 " learning period ends", color=TEAL, fontsize=9)
    axes[2].set_xlabel("date")
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_model_vs_protocol(rows, out_name="fig_model_vs_protocol.png"):
    """Does model capacity rescue a leaking protocol?

    `rows` is a list of dicts with keys: model, random, random_sd,
    grouped, grouped_sd.

    The message is carried by the *gap* between the two bars, repeated across
    very different model families. If a small linear model and a CNN both
    drop by a similar amount when the split stops leaking, the drop belongs
    to the protocol, not to the model.
    """
    def _wrap(t):
        """Break a two-word label onto two lines so axis labels stay readable."""
        parts = str(t).split()
        return "\n".join(parts) if len(parts) == 2 else str(t)

    labels = [_wrap(r["model"]) for r in rows]
    rand = [r["random"] for r in rows]
    rand_sd = [r.get("random_sd", 0.0) for r in rows]
    grp = [r["grouped"] for r in rows]
    grp_sd = [r.get("grouped_sd", 0.0) for r in rows]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(max(8.0, 2.2 * len(labels)), 5.0))

    ax.bar(x - w / 2, rand, w, yerr=rand_sd, capsize=4, color=GREY,
           label="Random split  (clips shuffled — literature protocol)",
           error_kw={"ecolor": "#5E7370", "lw": 1.1})
    ax.bar(x + w / 2, grp, w, yerr=grp_sd, capsize=4, color=MID,
           label="Grouped split  (test trees unseen)",
           error_kw={"ecolor": TEAL, "lw": 1.1})

    ax.axhline(0.5, color=ALERT, ls=":", lw=1.4, zorder=0)
    ax.text(-0.46, 0.515, "chance", color=ALERT, fontsize=9)

    for xi in range(len(labels)):
        ax.text(xi - w / 2, rand[xi] + rand_sd[xi] + 0.03, f"{rand[xi]:.2f}",
                ha="center", fontsize=9.5, color="#5E7370")
        ax.text(xi + w / 2, grp[xi] + grp_sd[xi] + 0.03, f"{grp[xi]:.2f}",
                ha="center", fontsize=9.5, fontweight="bold", color=TEAL)
        drop = rand[xi] - grp[xi]
        ax.annotate("", xy=(xi + w / 2, grp[xi]), xytext=(xi - w / 2, rand[xi]),
                    arrowprops=dict(arrowstyle="->", color=ALERT, lw=1.3,
                                    alpha=0.75))
        ax.text(xi, max(rand[xi], grp[xi]) + 0.11, f"−{drop:.2f}",
                ha="center", fontsize=10, fontweight="bold", color=ALERT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.3)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("balanced accuracy")
    ax.set_title("More model capacity does not survive an honest split",
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    ax.legend(frameon=False, loc="upper center", fontsize=9.5,
              bbox_to_anchor=(0.5, -0.09), ncol=1)
    ax.grid(axis="y", color="#E3EAE7", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_spectrogram_examples(images, labels, out_name="fig_spectrograms.png",
                             n_each: int = 4):
    """Example spectrogram images, as the published CNN pipelines see them."""
    fig, axes = plt.subplots(2, n_each, figsize=(2.3 * n_each, 5.0))
    for row, (lab, name) in enumerate([(0, "clean"), (1, "infested")]):
        idx = np.where(labels == lab)[0][:n_each]
        for col, i in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(images[i, 0], aspect="auto", origin="lower", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=11, fontweight="bold",
                              color=MID if lab == 0 else ALERT)
    fig.suptitle("What the CNN sees: log-mel spectrogram images",
                 fontsize=12, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = RESULTS_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
