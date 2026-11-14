"""
Project-wide configuration and constants
Dataset: https://www.kaggle.com/datasets/disbeat/bciaut-p300
    - 15 subjects (ASD), 7 sessions, Train/Test split per session
    - 8 EEG channels, 250 Hz sampling rate
    - Epochs: -200 ms - 1200 ms relative to stimulus onset (350 samples)
    - P300 oddball paradigm with 8 stimulus classes; target vs non-target labels
"""

from pathlib import Path


# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"

# Dataset parameters
SUBJECTS = [f"SBJ{i:02d}" for i in range(1, 16)]
SESSIONS = [f"S{i:02d}" for i in range(1, 8)]
N_SUBJECTS = 15
N_SESSIONS = 7

# EEG acquisition
SFREQ = 250#Hz
CHANNEL_NAMES = ["C3", "Cz", "C4", "CPz", "P3", "Pz", "P4", "POz"]
N_CHANNELS = len(CHANNEL_NAMES)
CHANNEL_TYPES = ["eeg"] * N_CHANNELS
TMIN = -0.200  # s
TMAX = 1.196   # s (last sample: TMIN + (N_SAMPLES - 1) / SFREQ)
N_SAMPLES = 350  # per epoch

# Stimulus
N_STIMULUS_CLASSES = 8# flash groups 1–8
TARGET_LABEL = 1
NON_TARGET_LABEL = 0

# Pipeline defaults 
XDAWN_N_COMPONENTS = 5
BANDPASS_LOW = 0.5   # Hz
BANDPASS_HIGH = 30.0  # Hz
RANDOM_SEED = 42
