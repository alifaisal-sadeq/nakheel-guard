# Abstract



Ali Faisal Sadeq Ahmed · Ismail Mudhaffar
King Fahd University of Petroleum and Minerals
SAIF 2026 — Team SAIF130280



Red palm weevil destroys date palms from the inside, and by the time crown collapse is visible the tree is usually beyond saving. Acoustic sensors that listen for larval feeding are the leading early-detection approach, and published studies report accuracies approaching 100%. This project tested whether those numbers describe performance on a tree the system has never heard before.

We used TreeVibes (Rigakis et al., 2021), the largest open dataset of field recordings from instrumented trees: 4,939 labelled recordings from 38 sensors across 142 days, with labels cross-checked against the dataset's own folder structure at 3,285 of 3,285 agreement. From a class-balanced subsample of 2,400 recordings we extracted 261 acoustic features across 38,675 analysis windows from 36 trees.

We then compared two evaluation protocols on identical features and identical models. Splitting recordings at random, the standard practice, gave 0.890 balanced accuracy. Splitting so that no tree appears in both training and test gave 0.485, against a chance level of 0.500. The gap is 0.405, and it reproduces across three model families: a spectrogram CNN scores highest under the random split at 0.917 and shows the widest gap of all at 0.434, so greater model capacity makes the problem worse rather than better.

The cause is measurable. A classifier trained on the same features predicted which of the 36 trees a recording came from with 0.868 accuracy against a 0.028 chance baseline — about 31 times chance. On unseen trees the random forest reached recall 0.923 with specificity 0.041, labelling almost every tree infested. These models learn sensor and tree identity rather than infestation.

We propose per-tree sequential monitoring as the alternative: a device models its own tree's healthy baseline and accumulates evidence of departure over time, rather than judging single recordings against other trees. At a matched false-alarm budget this detects more infestations, and detects them sooner, than per-clip classification.

## Limitations

TreeVibes records *Xylotrechus chinensis* in mulberry, not red palm weevil in date palm. It is used because it is the only large open dataset of field borer acoustics in which every recording is tagged to an individual tree, and that per-tree structure is what the question requires. The finding concerns evaluation protocol rather than species, so it should apply wherever recordings are split at random — but for red palm weevil specifically that is a prediction, not a measurement.

No tree in the dataset is recorded in both states, so tree identity and class are perfectly confounded and the per-tree method could not be fully validated; its shift direction retains only 10% of its training effect on unseen trees. No tree has 30 days of recordings, so a planned analysis of temperature-driven seasonal drift could not run, and that null is reported rather than dropped. The detection comparison runs on simulated streams drawn from real held-out score pools, so the comparison between detectors holds but the absolute rates and delays are not field performance. About 2% of audio files failed to decode. Nothing in this project has been validated on a living palm.

---

