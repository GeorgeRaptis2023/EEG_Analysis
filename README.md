## Pipeline Overview

1. **Load data** (`data_loading.py`) 
2. **Preprocess** (`preprocessing.py`)
3. **Extract features** (`feature_extraction.py`)
4. **Classify** (`classification.py`) 
5. **Evaluate** (`evaluation.py`)
6. **Visualize** (`visualization.py`)

## Implemented Components

* **Data loading:** Utilities for loading and structuring multi-subject, multi-session EEG datasets and event annotations.
* **Preprocessing:**
  * Bandpass filtering (0.5–30 Hz via zero-phase IIR Butterworth filter)
  * Baseline correction
  * Peak-to-peak artifact rejection with optional thresholding
  * Session- and subject-level preprocessing runners
* **Feature extraction:** Spatial filtering via xDAWN and downstream feature vector construction for the LDA classifier.
