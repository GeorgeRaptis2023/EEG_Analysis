"""
Data loading utilities
Handles reading the .mat epoch arrays, event/target/label text files
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import mne

from config import (
    DATA_DIR, SFREQ, CHANNEL_NAMES, CHANNEL_TYPES,
    TMIN, N_CHANNELS, N_SAMPLES, N_STIMULUS_CLASSES,
    SUBJECTS, SESSIONS
)

logger = logging.getLogger(__name__)
_PREFIX = {"Train": "train", "Test": "test"}

def load_mat(filepath):
    """Load an EEG .mat file and return the data array.
    The .mat file contains trainData or testData
    dtype float64 (n_channels, n_samples, n_epochs)."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    mat = scipy.io.loadmat(str(filepath))

    for key in ("trainData", "testData"):
        if key in mat:
            return mat[key]

    raise KeyError(f"No data found in {filepath}")

def load_runs_per_block(filepath):
    """Load the number of flash repetitions per block (Test split only)."""
    return int(np.loadtxt(str(filepath), dtype=int))

def load_session(subject, session, split="Train"):
    #Load all data for a single subject / session / split.
    d = DATA_DIR / subject / session / split
    if not d.exists():
        raise FileNotFoundError(f"Session directory not found: {d}")

    prefix = _PREFIX[split]

    result = {
        "data": np.loadtxt(str(d / f"{prefix}Data.mat"), dtype=int).ravel(),
        "events":  np.loadtxt(str(d / f"{prefix}Events.txt"), dtype=int).ravel() ,
        "targets": np.loadtxt(str(d / f"{prefix}Targets.txt"), dtype=int).ravel(),
        "labels": np.loadtxt(str(d / f"{prefix}Labels.txt"), dtype=int).ravel(),
        "subject": subject,
        "session": session,
        "split":   split,
        "runs_per_block": None
    }

    # runs_per_block.txt only in the Test split.
    rpb_path = d / "runs_per_block.txt"
    if rpb_path.exists():result["runs_per_block"] = load_runs_per_block(d/"runs_per_block.txt")

    n_epochs = result["data"].shape[2]
    assert result["events"].shape[0] == n_epochs, "Event count mismatch"
    assert result["targets"].shape[0] == n_epochs, "Target count mismatch"

    logger.info(
        "Loaded %s/%s/%s — %d epochs, %d blocks",
        subject, session, split, n_epochs, len(result["labels"]),
    )
    return result

def load_subject(subject, split="Train", sessions=None):
    #Concatenate sessions for one subject.

    sessions = sessions or SESSIONS
    all_data, all_events, all_targets, all_labels = [],[],[],[]
    all_session_ids = []
    all_rpb = []

    for ses in sessions:
        sess = load_session(subject, ses, split)
        n_ep = sess["data"].shape[2]

        all_data.append(sess["data"])
        all_events.append(sess["events"])
        all_targets.append(sess["targets"])
        all_labels.append(sess["labels"])
        all_session_ids.extend([ses] * n_ep)
        all_rpb.append(sess["runs_per_block"])

    return {
        "data":    np.concatenate(all_data, axis=2),
        "events":  np.concatenate(all_events),
        "targets": np.concatenate(all_targets),
        "labels":  np.concatenate(all_labels),
        "subject": subject,
        "session": np.array(all_session_ids),
        "split":   split,
        "runs_per_block": all_rpb,
    }

def load_all_subjects(split="Train", sessions=None):
    for subj in SUBJECTS:
        yield subj, load_subject(subj, split=split, sessions=sessions)

def mat_to_mne_epochs(data, targets, events=None):
    epoch_data = np.transpose(data, (2, 0, 1)).astype(np.float64)
    n_epochs = epoch_data.shape[0]

    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=SFREQ, ch_types=CHANNEL_TYPES)
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, match_case=False)

    mne_events = np.column_stack([
        np.arange(n_epochs),           # sample index 
        np.zeros(n_epochs, dtype=int), # unused
        targets.astype(int),           # event id = target flag
    ])

    event_id = {"non_target": 0, "target": 1}

    epochs = mne.EpochsArray(
        epoch_data,
        info,
        events=mne_events,
        tmin=TMIN,
        event_id=event_id,
        verbose=False,
    )

    # Attached for character-level decoding.
    if events is not None:
        metadata = pd.DataFrame({
            "stimulus_id": events.astype(int),
            "target": targets.astype(int),
        })
        epochs.metadata = metadata

    return epochs

def load_session_as_epochs(subject, session, split="Train"):
    sess = load_session(subject, session, split)
    return mat_to_mne_epochs(sess["data"], sess["targets"], sess["events"])
