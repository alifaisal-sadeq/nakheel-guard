"""Central configuration for the Nakheel Guard acoustic detection pipeline."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_PATH = RESULTS_DIR / "model.joblib"

# ---------------------------------------------------------------- audio
# Red Palm Weevil larvae produce short, broadband feeding impulses whose
# energy is concentrated at low frequencies. Wang et al. (2021) report the
# bulk of larval acoustic power near 400 Hz; Ashry et al. (2022) band-pass
# their DAS signals to 100-800 Hz. We keep a slightly wider band so that
# higher-frequency chewing transients survive, then resample low to keep
# feature extraction fast enough to run on a laptop.
SAMPLE_RATE = 8000
BAND_LOW_HZ = 150
BAND_HIGH_HZ = 3000

# Each source recording is cut into fixed-length analysis windows. Shorter
# windows give more training examples and match what a field device would
# transmit; too short and a sparse feeding burst falls outside the window.
WINDOW_SECONDS = 2.0
WINDOW_HOP_SECONDS = 1.0
MIN_WINDOW_SECONDS = 1.0

# ---------------------------------------------------------------- features
N_MELS = 40
N_MFCC = 20
N_FFT = 512
HOP_LENGTH = 128

# ---------------------------------------------------------------- evaluation
RANDOM_SEED = 42
N_SPLITS = 5

# Decision threshold is tuned on validation data rather than left at 0.5,
# because missing an infested tree costs far more than a false alarm:
# a false alarm costs one manual inspection, a miss costs the tree.
FALSE_ALARM_BUDGET = 0.10  # max acceptable false-positive rate on clean trees

LABELS = {"clean": 0, "infested": 1}
LABEL_NAMES = {0: "clean", 1: "infested"}
