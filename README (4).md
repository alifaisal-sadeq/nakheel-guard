# Nakheel Guard

**Acoustic detection of wood-boring pests reports 0.89 balanced accuracy and delivers 0.48 on a tree it has not heard before.**

Findings page: **https://alifaisal-sadeq.github.io/nakheel-guard/**
Abstract: **[ABSTRACT.md](ABSTRACT.md)**

SAIF 2026 (Tuwaiq Academy), Team SAIF130280. Ali Faisal Sadeq Ahmed and Ismail Mudhaffar, King Fahd University of Petroleum and Minerals.

---

## The finding

Identical features. Identical models. The only thing that changes is whether a tree is allowed to appear in both the training set and the test set.

| Model | Random split | Grouped split (unseen trees) | Gap |
|---|---|---|---|
| Logistic regression | 0.890 | 0.485 | **+0.405** |
| Random forest | 0.854 | 0.482 | **+0.373** |
| Spectrogram CNN | 0.917 | 0.483 | **+0.434** |

Balanced accuracy, five-fold cross-validation, 38,675 analysis windows across 36 trees. Chance is 0.500.

The random split is standard practice in the published literature. The grouped split uses `StratifiedGroupKFold` with the tree as the group, so no tree is in both sets.

The strongest model has the widest gap. More capacity makes the problem worse, not better.

## The cause

A gap alone does not say what leaked. The same features were used to train a classifier on a different target — not *is this infested*, but *which of the 36 trees is this*.

It succeeds at **0.868** against a chance level of **0.028**: about 31 times chance. Tree identity is present in the features in a form any model can exploit, and a random split rewards exploiting it.

On unseen trees the random forest reaches recall 0.923 with specificity 0.041 — it labels almost every tree infested.

## The alternative

A sensor bolted to one trunk can ask a question a roving probe cannot: *has this tree changed from its own baseline?* Tested with a CUSUM sequential detector (Page, 1954) against per-clip classification, both first calibrated to the same false-alarm budget of 0.5 false alarms per tree per 180 observations.

| Prevalence | Sequential rate | Delay | Per-clip rate | Delay |
|---|---|---|---|---|
| 10% | 0.700 | 38 | 0.437 | 51 |
| 20% | 0.797 | 29 | 0.523 | 52 |
| 35% | 0.853 | 19 | 0.633 | 40 |
| 50% | 0.807 | 14 | 0.700 | 36 |

## Stated limitations

- **This is not red palm weevil.** TreeVibes records *Xylotrechus chinensis* in mulberry. It was used because it is the only large open dataset of field borer acoustics in which every recording is tagged to an individual tree. The finding concerns evaluation protocol rather than species, so it should transfer — but for red palm weevil that is a prediction, not a measurement.
- **No field validation.** Nothing here has touched a living palm, and no real-world false-alarm rate is claimed.
- **The detection comparison runs on simulated streams** drawn from real held-out score pools. The relative comparison between detectors holds; the absolute rates and delays are not field performance.
- **The per-tree method could not be fully validated.** No tree in TreeVibes is recorded both healthy and infested, so tree identity and class are perfectly confounded. The shift direction retains only 10% of its training effect on unseen trees (Cohen's *d* falls from +3.35 to +0.33).
- **The seasonal analysis returned nothing.** No tree has 30 days of recordings, so the temperature-confound test could not run. Reported here rather than dropped.
- **About 2% of audio files failed to decode** — roughly 115 Ogg files needing a newer codec build.

## Data

[TreeVibes](https://www.kaggle.com/datasets/potamitis/treevibes) — Rigakis et al. (2021), published openly under CC BY. Not included in this repository; download it from Kaggle.

4,939 labelled field recordings from 38 sensor-equipped trees across 142 days. The analysis uses a class-balanced subsample of 2,400 recordings.

Labels were cross-checked against the dataset's own `train/clean` and `train/infested` folders: 3,285 of 3,285 agree. The annotation table has two columns that could plausibly be the label, and using the wrong one produces a near-total inversion (68 clean against 4,871 infested), so this check is not optional.

## References

- Rigakis, I., Potamitis, I., Tatlas, N.-A., Potirakis, S. M., & Ntalampiras, S. (2021). TreeVibes: Modern tools for global monitoring of trees for borers. *Smart Cities*, 4(1), 271–285.
- Boulila, W., Driss, M., Al-Sarem, M., Saeed, F., & Krichen, M. (2023). Early detection of red palm weevil infestations using deep learning classification of acoustic signals. *Computers and Electronics in Agriculture*, 212, 108154.
- Page, E. S. (1954). Continuous inspection schemes. *Biometrika*, 41(1–2), 100–115.
- Systematic review of acoustic detection of wood-boring pests (2026). *Insects*, 17(10), 1081.
