## Pipeline Overview

1. **Load data** (`data_loading.py`) — Loads raw EEG recordings, channel metadata, and event markers across sessions and subjects.
2. **Preprocess** (`preprocessing.py`) — Cleans raw signals, applies filtering, extracts epochs, and rejects noisy trials.
3. **Extract features** (`feature_extraction.py`) — Applies spatial filtering and constructs discriminative feature vectors from preprocessed epochs.
4. **Classify** (`classification.py`) — Trains and runs the Linear Discriminant Analysis (LDA) classifier on extracted features.
5. **Evaluate** (`evaluation.py`) — Computes classification performance metrics across trials, sessions, and subjects.
6. **Visualize** (`visualization.py`) — Generates ERP waveform plots, spatial filter topographies, and classification performance figures.

## Implemented Components

* **Data loading:** Utilities for loading and structuring multi-subject, multi-session EEG datasets and event annotations.
* **Preprocessing:**
  * Bandpass filtering (0.5–30 Hz via zero-phase IIR Butterworth filter)
  * Baseline correction
  * Peak-to-peak artifact rejection with optional thresholding
  * Session- and subject-level preprocessing runners
* **Feature extraction:** Spatial filtering via xDAWN and downstream feature vector construction for the LDA classifier.
