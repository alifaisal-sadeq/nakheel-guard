"""Audio feature extraction for Red Palm Weevil acoustic detection.

Three families of features are computed per analysis window:

1. Spectral envelope  - log-mel band energies and MFCCs. Standard for audio
                        classification; captures the overall timbre of the
                        signal inside the trunk.
2. Spectral shape     - centroid, bandwidth, roll-off, flatness, zero-crossing
                        rate. Separates tonal background (wind, machinery)
                        from broadband transients.
3. Impulse statistics - larvae do not produce a continuous sound. They chew in
                        short discrete bursts. Counting those bursts and
                        measuring their spacing is the domain-specific signal
                        that generic audio features miss, and it is what human
                        listeners are actually attending to when they inspect a
                        palm with a probe and headphones.

Feature 3 is the one that carries the biology. It is implemented here rather
than taken off the shelf.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt, find_peaks

from config import (
    SAMPLE_RATE, BAND_LOW_HZ, BAND_HIGH_HZ,
    N_MELS, N_MFCC, N_FFT, HOP_LENGTH,
    WINDOW_SECONDS, WINDOW_HOP_SECONDS, MIN_WINDOW_SECONDS,
)


# --------------------------------------------------------------------------
# signal conditioning
# --------------------------------------------------------------------------

def _bandpass_sos(sr: int):
    nyq = sr / 2.0
    low = max(BAND_LOW_HZ / nyq, 1e-4)
    high = min(BAND_HIGH_HZ / nyq, 0.999)
    return butter(4, [low, high], btype="band", output="sos")


def load_audio(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a file as mono float32 at the target rate, band-passed."""
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        return y
    y = y - np.mean(y)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    if y.size > 27:  # sosfiltfilt needs enough samples for its padding
        y = sosfiltfilt(_bandpass_sos(sr), y).astype(np.float32)
    return y.astype(np.float32)


def frame_windows(y: np.ndarray, sr: int = SAMPLE_RATE):
    """Split a recording into overlapping fixed-length analysis windows."""
    win = int(WINDOW_SECONDS * sr)
    hop = int(WINDOW_HOP_SECONDS * sr)
    min_len = int(MIN_WINDOW_SECONDS * sr)
    if y.size < min_len:
        return []
    if y.size <= win:
        return [y]
    starts = range(0, y.size - win + 1, hop)
    return [y[s:s + win] for s in starts]


# --------------------------------------------------------------------------
# impulse / burst statistics  (the domain-specific part)
# --------------------------------------------------------------------------

def burst_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Detect short energy bursts and describe their rate and spacing.

    A feeding larva produces trains of brief impulses separated by irregular
    gaps. Wind and machinery produce either continuous energy or very few,
    long events. The statistics below are designed to tell those apart
    without needing to hear the sound.
    """
    if y.size < 64:
        return {f"burst_{k}": 0.0 for k in
                ("rate", "mean_amp", "max_amp", "mean_dur", "ibi_mean",
                 "ibi_std", "ibi_cv", "crest", "amp_std")}

    # Short-time energy envelope, in dB relative to the window median.
    # 12 ms frames, 6 ms hop. Larval feeding impulses last roughly 5-30 ms,
    # so this resolves them while smoothing the frame-to-frame ripple of
    # filtered noise, which otherwise generates spurious peaks faster than
    # the real events occur.
    frame = max(64, int(0.012 * sr))
    hop = max(32, frame // 2)
    env = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    env_db = librosa.amplitude_to_db(env + 1e-10)

    median = np.median(env_db)
    mad = np.median(np.abs(env_db - median)) + 1e-9

    # Detect impulses by PROMINENCE (how far a peak rises above the local
    # terrain) rather than by absolute height.
    #
    # Absolute thresholding fails here in both directions. The envelope is in
    # dB, which compresses dynamic range to ~10-15 dB, so a threshold set a
    # few MAD above the median can exceed the signal maximum and detect
    # nothing. And when feeding is dense the impulses overlap and lift the
    # noise floor itself, so the busiest recordings -- the ones that most
    # clearly indicate infestation -- have the least headroom above their own
    # median. Prominence is immune to both: it measures local rise, so it
    # behaves the same whether the floor is quiet or crowded.
    frames_per_sec = sr / hop
    min_distance = max(1, int(0.004 * frames_per_sec))  # 4 ms refractory
    # Threshold in absolute dB, which is physically interpretable: 6 dB of
    # prominence is a doubling of local amplitude. A MAD-scaled floor keeps
    # it from firing on envelope ripple in an unusually noisy recording.
    min_prominence = max(6.0, 2.5 * mad)
    peaks, props = find_peaks(env_db, prominence=min_prominence,
                              distance=min_distance)

    duration_s = y.size / sr
    if peaks.size == 0:
        return {
            "burst_rate": 0.0, "burst_mean_amp": 0.0, "burst_max_amp": 0.0,
            "burst_mean_dur": 0.0, "burst_ibi_mean": 0.0, "burst_ibi_std": 0.0,
            "burst_ibi_cv": 0.0,
            "burst_crest": float(np.max(env) / (np.mean(env) + 1e-10)),
            "burst_amp_std": float(np.std(env_db)),
        }

    heights = props["prominences"]                    # local rise, dB
    ibi = np.diff(peaks) / frames_per_sec if peaks.size > 1 else np.array([0.0])

    # Width of each burst at half prominence, in milliseconds.
    try:
        from scipy.signal import peak_widths
        widths = peak_widths(env_db, peaks, rel_height=0.5)[0] / frames_per_sec
        mean_dur = float(np.mean(widths) * 1000.0)
    except Exception:
        mean_dur = 0.0

    return {
        "burst_rate": float(peaks.size / duration_s),
        "burst_mean_amp": float(np.mean(heights)),
        "burst_max_amp": float(np.max(heights)),
        "burst_mean_dur": mean_dur,
        "burst_ibi_mean": float(np.mean(ibi)),
        "burst_ibi_std": float(np.std(ibi)),
        # Coefficient of variation of inter-burst intervals: regular machinery
        # noise gives a low value, irregular biological chewing a high one.
        "burst_ibi_cv": float(np.std(ibi) / (np.mean(ibi) + 1e-9)),
        "burst_crest": float(np.max(env) / (np.mean(env) + 1e-10)),
        "burst_amp_std": float(np.std(env_db)),
    }


# --------------------------------------------------------------------------
# full feature vector
# --------------------------------------------------------------------------

def _stats(mat: np.ndarray, prefix: str) -> dict:
    """Summarise a (bands x frames) matrix over time."""
    out = {}
    for name, fn in (("mean", np.mean), ("std", np.std), ("max", np.max)):
        vals = fn(mat, axis=1)
        for i, v in enumerate(vals):
            out[f"{prefix}_{name}_{i}"] = float(v)
    return out


def window_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Full feature dictionary for one analysis window."""
    feats: dict = {}

    n_fft = min(N_FFT, max(64, 2 ** int(np.floor(np.log2(max(y.size, 64))))))
    hop = max(16, min(HOP_LENGTH, n_fft // 4))

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop,
        n_mels=N_MELS, fmin=BAND_LOW_HZ, fmax=min(BAND_HIGH_HZ, sr // 2 - 1),
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    feats.update(_stats(log_mel, "mel"))

    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=N_MFCC)
    feats.update(_stats(mfcc, "mfcc"))
    if mfcc.shape[1] >= 9:
        feats.update(_stats(librosa.feature.delta(mfcc), "dmfcc"))
    else:
        for name in ("mean", "std", "max"):
            for i in range(N_MFCC):
                feats[f"dmfcc_{name}_{i}"] = 0.0

    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    scalar_maps = {
        "centroid": librosa.feature.spectral_centroid(S=S, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=S, sr=sr),
        "flatness": librosa.feature.spectral_flatness(S=S),
        "zcr": librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop),
        "rms": librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop),
    }
    for name, mat in scalar_maps.items():
        v = mat[0]
        feats[f"{name}_mean"] = float(np.mean(v))
        feats[f"{name}_std"] = float(np.std(v))

    feats.update(burst_features(y, sr))
    return feats


def feature_names(sr: int = SAMPLE_RATE) -> list:
    """Stable ordered list of feature names (derived from a dummy window)."""
    dummy = np.random.RandomState(0).randn(int(WINDOW_SECONDS * sr)).astype(np.float32) * 1e-3
    return sorted(window_features(dummy, sr).keys())
