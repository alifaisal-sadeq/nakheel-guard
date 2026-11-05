"""The published approach, reimplemented: spectrogram images into a CNN.

Why this module exists
----------------------
Boulila et al. (2023) convert each recording into cepstral-coefficient images,
stack them, and classify with ImageNet CNNs (ResNet, VGG, MobileNet),
reporting 100% accuracy -- including on TreeVibes. Karar et al. (2022) do the
same with MixConvNet and report 97.4%.

This project classifies hand-designed features instead. That invites an
obvious question: *did you just not try deep learning?*

So we try it, under the same two evaluation protocols used everywhere else in
this project. The purpose is not to win a benchmark. It is to test whether
model capacity has anything to do with the problem:

  - under a RANDOM split, does a CNN reproduce the published near-perfect
    numbers?
  - under a GROUPED split, where test trees are unseen, does it survive?

If the CNN scores near 100% on random and collapses on grouped, then the
published results measure the protocol rather than the method, and no amount
of architecture will fix it. That is a stronger statement than any accuracy
figure this project could report on its own.

Design notes
------------
The CNN here is deliberately small (~100k parameters, three convolutional
blocks). That is not a handicap -- it is the honest choice. The number of
*independent* samples in this data is the number of trees (roughly 10-35),
not the number of clips. A 25-million-parameter ImageNet backbone fine-tuned
on 35 independent units will fit the trees perfectly and tell you nothing,
which is precisely the failure this project is documenting. A small model
makes the comparison about the protocol rather than about overfitting
capacity, and it runs on a CPU.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = object

from config import SAMPLE_RATE, BAND_LOW_HZ, BAND_HIGH_HZ, RANDOM_SEED


# --------------------------------------------------------------------------
# audio -> image
# --------------------------------------------------------------------------

def spectrogram_image(y, sr: int = SAMPLE_RATE, n_mels: int = 48,
                      n_frames: int = 48) -> np.ndarray:
    """One analysis window as a fixed-size log-mel image, scaled to [0, 1].

    Per-image normalisation (rather than a dataset-wide scale) matches what
    the published pipelines do when they save spectrograms as PNGs, and it
    removes absolute gain differences between devices.
    """
    import librosa

    n_fft = 512
    hop = max(1, len(y) // n_frames)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels,
        fmin=BAND_LOW_HZ, fmax=min(BAND_HIGH_HZ, sr // 2 - 1))
    img = librosa.power_to_db(mel + 1e-10)

    # Pad or crop to a fixed width.
    if img.shape[1] < n_frames:
        img = np.pad(img, ((0, 0), (0, n_frames - img.shape[1])),
                     mode="edge")
    img = img[:, :n_frames]

    lo, hi = img.min(), img.max()
    img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)
    return img.astype(np.float32)


def build_image_set(paths, labels, groups, n_mels: int = 48,
                    n_frames: int = 48, verbose: bool = True):
    """Turn a list of recordings into a stack of spectrogram images."""
    from features import load_audio, frame_windows

    imgs, ys, gs, ids = [], [], [], []
    for i, (p, lab, g) in enumerate(zip(paths, labels, groups)):
        try:
            audio = load_audio(p)
            for w in frame_windows(audio):
                imgs.append(spectrogram_image(w, n_mels=n_mels, n_frames=n_frames))
                ys.append(int(lab))
                gs.append(g)
                ids.append(p)
        except Exception:
            continue
        if verbose and (i + 1) % 300 == 0:
            print(f"    {i + 1}/{len(paths)} recordings", flush=True)

    if not imgs:
        raise RuntimeError("No spectrogram images were produced.")
    X = np.stack(imgs)[:, None, :, :]          # (N, 1, mels, frames)
    return X, np.array(ys), np.array(gs), np.array(ids)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class SmallCNN(nn.Module):
        """Three conv blocks, global pooling, linear head. ~100k parameters."""

        def __init__(self, n_mels: int = 48, n_frames: int = 48):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 2))

        def forward(self, x):
            return self.head(self.features(x))


def train_cnn(X_tr, y_tr, X_eval_list, epochs: int = 12, batch_size: int = 64,
              lr: float = 1e-3, seed: int = RANDOM_SEED, verbose: bool = False):
    """Train the CNN once and score every array in `X_eval_list`.

    Returns a list of P(infested) arrays, one per input. Training once and
    scoring both the training fold (to set the decision threshold) and the
    test fold keeps the threshold consistent with the model that produced it,
    and halves the compute.

    Class weighting matters as much here as for the feature models: without
    it an imbalanced fold produces a network that predicts the majority class
    and reports a respectable accuracy for doing nothing.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. pip install torch")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)

    model = SmallCNN(X_tr.shape[2], X_tr.shape[3])
    counts = np.bincount(y_tr, minlength=2).astype(float)
    weights = torch.tensor((counts.sum() / (2 * np.maximum(counts, 1))),
                           dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    n = len(yt)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = criterion(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if verbose:
            print(f"      epoch {ep + 1}/{epochs}  loss {total / n:.4f}")

    model.eval()
    results = []
    with torch.no_grad():
        for X_eval in X_eval_list:
            Xe = torch.tensor(X_eval, dtype=torch.float32)
            outs = []
            for i in range(0, len(Xe), 256):
                outs.append(torch.softmax(model(Xe[i:i + 256]), dim=1)[:, 1].numpy())
            results.append(np.concatenate(outs))
    return results


# --------------------------------------------------------------------------
# evaluation under both protocols
# --------------------------------------------------------------------------

def evaluate(X_img, y, groups, protocol: str, n_splits: int = 5,
             epochs: int = 12, verbose: bool = True) -> dict:
    """Cross-validate the CNN under 'grouped' or 'random' splits.

    Uses exactly the same splitters as the feature models, so the comparison
    isolates the model and the protocol and nothing else.
    """
    import pandas as pd
    from sklearn.metrics import (
        balanced_accuracy_score, recall_score, roc_auc_score,
        average_precision_score, confusion_matrix, precision_score, f1_score)
    import data as data_mod
    from train import threshold_for_false_alarm_budget

    df = pd.DataFrame({"y": y, "group": groups})
    splits = list(data_mod.grouped_splits(df, n_splits) if protocol == "grouped"
                  else data_mod.random_splits(df, n_splits))

    rows = []
    for k, (tr, te) in enumerate(splits):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        if verbose:
            print(f"    fold {k + 1}/{len(splits)} "
                  f"(train {len(tr)}, test {len(te)})", flush=True)

        train_scores, scores = train_cnn(
            X_img[tr], y[tr], [X_img[tr], X_img[te]], epochs=epochs)
        thr = threshold_for_false_alarm_budget(y[tr], train_scores)
        pred = (scores >= thr).astype(int)

        tn, fp, fn, tp = confusion_matrix(y[te], pred, labels=[0, 1]).ravel()
        rows.append({
            "fold": k,
            "balanced_accuracy": balanced_accuracy_score(y[te], pred),
            "recall": recall_score(y[te], pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "precision": precision_score(y[te], pred, zero_division=0),
            "f1": f1_score(y[te], pred, zero_division=0),
            "roc_auc": roc_auc_score(y[te], scores),
            "pr_auc": average_precision_score(y[te], scores),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })

    if not rows:
        raise RuntimeError(f"No usable folds for protocol '{protocol}'.")

    folds = pd.DataFrame(rows)
    numeric = folds.select_dtypes(include=[np.number])
    return {"folds": folds.to_dict(orient="records"),
            "mean": numeric.mean().to_dict(),
            "std": numeric.std(ddof=0).to_dict()}
