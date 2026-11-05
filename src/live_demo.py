"""Live demonstration for the exhibition booth.

Listens to a microphone (or replays an audio file) and classifies the signal
in real time using the trained model, showing a waveform, a spectrogram, a
probability trace and a large status readout.

    # live from the laptop microphone (needs: pip install sounddevice)
    python src/live_demo.py

    # replay a file, for rehearsal or when the booth is noisy
    python src/live_demo.py --file data/synthetic/tree_00/infested/clip_000.wav

    # fullscreen for the booth
    python src/live_demo.py --fullscreen

Press q or Escape to quit.

Booth note: a laptop's built-in microphone picks up the hall, not a tree.
For a convincing demo either replay recorded clips with --file, or plug in a
contact microphone and hold it against a wooden object while tapping it.
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import SAMPLE_RATE, WINDOW_SECONDS, MODEL_PATH
from features import load_audio, window_features

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

TEAL = "#123F3B"
CREAM = "#F3F0E1"
MID = "#2E6B63"
ALERT = "#C0392B"
AMBER = "#E08A3C"


# --------------------------------------------------------------------------
# audio sources
# --------------------------------------------------------------------------

class MicSource:
    """Microphone input via sounddevice, exposed as a block queue."""

    def __init__(self, sr=SAMPLE_RATE, block=0.25):
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Microphone input needs the 'sounddevice' package.\n"
                "  pip install sounddevice\n"
                "Or replay a file instead:  python src/live_demo.py --file <path>"
            ) from exc
        self.sd = sd
        self.sr = sr
        self.blocksize = int(block * sr)
        self.q: queue.Queue = queue.Queue()
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream = self.sd.InputStream(
            samplerate=self.sr, channels=1, dtype="float32",
            blocksize=self.blocksize, callback=self._callback)
        self.stream.start()

    def read(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class FileSource:
    """Replay a file in real time, looping."""

    def __init__(self, path, sr=SAMPLE_RATE, block=0.25, loop=True):
        self.y = load_audio(path, sr)
        if self.y.size == 0:
            raise RuntimeError(f"Could not read any audio from {path}")
        self.sr = sr
        self.blocksize = int(block * sr)
        self.pos = 0
        self.loop = loop
        self._t0 = None

    def start(self):
        self._t0 = time.time()

    def read(self):
        if self.pos >= self.y.size:
            if not self.loop:
                return None
            self.pos = 0
        # Pace playback so the display moves at real speed.
        time.sleep(self.blocksize / self.sr)
        block = self.y[self.pos:self.pos + self.blocksize]
        self.pos += self.blocksize
        if block.size < self.blocksize:
            block = np.pad(block, (0, self.blocksize - block.size))
        return block.astype(np.float32)

    def stop(self):
        pass


# --------------------------------------------------------------------------
# classifier wrapper
# --------------------------------------------------------------------------

class Detector:
    def __init__(self, model_path=MODEL_PATH):
        import joblib
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"No trained model at {model_path}.\n"
                "Train one first:  python src/train.py --data <dataset>"
            )
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.names = bundle["feature_names"]
        self.model_name = bundle.get("model_name", "model")

    def score(self, window: np.ndarray) -> float:
        feats = window_features(window)
        x = np.array([[feats.get(n, 0.0) for n in self.names]], dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(x)[0, 1])
        return float(self.model.decision_function(x)[0])


# --------------------------------------------------------------------------
# display
# --------------------------------------------------------------------------

def run(source, detector, fullscreen=False, history_s=60):
    sr = SAMPLE_RATE
    win_len = int(WINDOW_SECONDS * sr)
    ring = deque(maxlen=win_len)
    scores = deque(maxlen=history_s)
    times = deque(maxlen=history_s)

    plt.rcParams["figure.facecolor"] = TEAL
    plt.rcParams["savefig.facecolor"] = TEAL
    fig = plt.figure(figsize=(13, 7.5))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[2.0, 1.0],
                  hspace=0.45, wspace=0.22,
                  left=0.06, right=0.97, top=0.88, bottom=0.08)

    fig.text(0.06, 0.945, "NAKHEEL GUARD", color=CREAM,
             fontsize=23, fontweight="bold")
    fig.text(0.06, 0.912, "Live acoustic detection of Red Palm Weevil",
             color="#9CBDB6", fontsize=12)
    fig.text(0.97, 0.945, detector.model_name.replace("_", " "),
             color="#6E8E88", fontsize=10, ha="right")

    ax_wave = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_hist = fig.add_subplot(gs[2, 0])
    ax_stat = fig.add_subplot(gs[:, 1])

    for ax in (ax_wave, ax_spec, ax_hist):
        ax.set_facecolor("#0E3330")
        for s in ax.spines.values():
            s.set_color("#2A514C")
        ax.tick_params(colors="#7FA39C", labelsize=8)
        ax.title.set_color(CREAM)

    ax_wave.set_title("Signal", fontsize=11, loc="left")
    ax_wave.set_ylim(-1, 1); ax_wave.set_xticks([])
    (wave_line,) = ax_wave.plot(np.zeros(win_len), color=MID, lw=0.7)

    ax_spec.set_title("Spectrogram", fontsize=11, loc="left")
    ax_spec.set_ylabel("Hz", color="#7FA39C", fontsize=9)
    spec_img = ax_spec.imshow(
        np.zeros((64, 80)), aspect="auto", origin="lower",
        cmap="magma", vmin=-70, vmax=10,
        extent=[0, 80, 0, sr / 2])
    ax_spec.set_xticks([])

    ax_hist.set_title("Infestation probability", fontsize=11, loc="left")
    ax_hist.set_ylim(0, 1); ax_hist.set_xlim(0, history_s)
    ax_hist.axhline(detector.threshold, color=AMBER, ls="--", lw=1.2)
    ax_hist.set_xticks([])
    (hist_line,) = ax_hist.plot([], [], color=CREAM, lw=1.8)

    ax_stat.set_facecolor("#0E3330")
    ax_stat.set_xticks([]); ax_stat.set_yticks([])
    for s in ax_stat.spines.values():
        s.set_color("#2A514C")
    status_txt = ax_stat.text(0.5, 0.62, "LISTENING", color=CREAM,
                              fontsize=30, fontweight="bold",
                              ha="center", va="center")
    prob_txt = ax_stat.text(0.5, 0.42, "--", color="#9CBDB6",
                            fontsize=52, fontweight="bold",
                            ha="center", va="center")
    sub_txt = ax_stat.text(0.5, 0.26, "probability", color="#6E8E88",
                           fontsize=11, ha="center", va="center")
    bar = ax_stat.barh([0.12], [0.0], height=0.055, left=0.08,
                       color=MID)[0]
    ax_stat.set_xlim(0, 1); ax_stat.set_ylim(0, 1)

    if fullscreen:
        try:
            fig.canvas.manager.full_screen_toggle()
        except Exception:
            pass

    stop_flag = threading.Event()

    def on_key(event):
        if event.key in ("q", "escape"):
            stop_flag.set()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: stop_flag.set())

    source.start()
    plt.show(block=False)
    t_start = time.time()
    last_score_t = 0.0

    try:
        while not stop_flag.is_set():
            block = source.read()
            if block is None:
                break
            ring.extend(block)
            if len(ring) < win_len:
                plt.pause(0.001)
                continue

            buf = np.asarray(ring, dtype=np.float32)
            now = time.time() - t_start

            # Classify about twice a second; feature extraction is the
            # expensive step and the display stays smooth regardless.
            if now - last_score_t >= 0.5:
                p = detector.score(buf)
                scores.append(p); times.append(now)
                last_score_t = now

                infested = p >= detector.threshold
                colour = ALERT if infested else MID
                status_txt.set_text("INFESTED" if infested else "HEALTHY")
                status_txt.set_color(ALERT if infested else CREAM)
                prob_txt.set_text(f"{p*100:.0f}%")
                prob_txt.set_color(colour)
                sub_txt.set_text("larval activity detected" if infested
                                 else "no larval activity")
                bar.set_width(max(0.01, p * 0.84))
                bar.set_color(colour)
                ax_stat.set_facecolor("#3A1512" if infested else "#0E3330")

                hist_line.set_data(np.arange(len(scores)), list(scores))
                ax_hist.set_xlim(0, max(history_s, len(scores)))

            wave_line.set_ydata(buf)
            lim = max(0.05, float(np.max(np.abs(buf))) * 1.1)
            ax_wave.set_ylim(-lim, lim)

            spec = np.abs(np.fft.rfft(
                buf.reshape(-1, 256)[: (buf.size // 256)] *
                np.hanning(256), axis=1))
            spec_db = 20 * np.log10(spec.T + 1e-6)
            spec_img.set_data(spec_db)
            spec_img.set_extent([0, spec_db.shape[1], 0, sr / 2])

            fig.canvas.draw_idle()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        plt.close("all")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="replay this audio file instead of the microphone")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--list-devices", action="store_true",
                    help="print available audio input devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return

    detector = Detector(args.model)
    source = FileSource(args.file) if args.file else MicSource()
    run(source, detector, fullscreen=args.fullscreen)


if __name__ == "__main__":
    main()
