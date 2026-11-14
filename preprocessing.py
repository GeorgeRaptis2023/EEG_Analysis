"""
Preprocessing pipeline for BCIAUT-P300 EEG data.

Pipeline stages:
    1. Bandpass filtering (0.5–30 Hz via zero-phase IIR Butterworth)
    2. Baseline correction (pre-stimulus window, -0.2 s to 0.0 s)
    3. Peak-to-peak artifact rejection (optional thresholding)
    4. Session & subject-level preprocessing runners
"""

import logging
import numpy as np
import mne

from config import BANDPASS_LOW, BANDPASS_HIGH, TMIN, SESSIONS
from data_loading import load_session, mat_to_mne_epochs

logger = logging.getLogger(__name__)


def bandpass_filter(epochs, low=BANDPASS_LOW, high=BANDPASS_HIGH):
    """Apply a zero-phase bandpass filter to MNE Epochs.
    Uses a 4th-order forward-backward Butterworth IIR filter
    """
    filtered = epochs.copy()
    filtered.filter(l_freq=low,h_freq=high,method="iir",
    iir_params=dict(order=4, ftype="butter"),verbose=False)
    return filtered


def baseline_correct(epochs, baseline=(TMIN, 0.0)):
    #Subtract the mean of the pre-stimulus baseline from each epoch.

    corrected = epochs.copy()
    corrected.apply_baseline(baseline=baseline, verbose=False)
    return corrected


def reject_bad_epochs(epochs, threshold_uv=100.0):
    #Reject epochs whose peak-to-peak amplitude exceeds a threshold.

    if threshold_uv is None:
        return epochs.copy(), np.arange(len(epochs)), np.array([], dtype=int)

    data = epochs.get_data()  #(n_epochs, n_channels, n_samples)
    ptp = np.ptp(data, axis=-1)
    max_ptp = ptp.max(axis=-1)

    good_indices = np.where(max_ptp <= threshold_uv)[0]
    bad_indices = np.where(max_ptp > threshold_uv)[0]

    cleaned_epochs = epochs.copy()[good_indices]
    return cleaned_epochs


def preprocess_session(data_dict,baseline=(TMIN, 0.0),low=BANDPASS_LOW,
                       high=BANDPASS_HIGH,threshold_uv=None):


    epochs = mat_to_mne_epochs(data_dict["data"],
    data_dict["targets"],data_dict["events"])

    epochs = bandpass_filter(epochs, low=low, high=high)
    epochs = baseline_correct(epochs, baseline=baseline)

    if threshold_uv is not None:
        epochs = reject_bad_epochs(epochs, threshold_uv=threshold_uv)

    return epochs


def preprocess_subject(subject,split="Train",sessions=None,low=BANDPASS_LOW,
                       high=BANDPASS_HIGH,baseline=(TMIN, 0.0),threshold_uv=None):
    if(sessions==None):sessions=SESSIONS

    session_epochs = []
    for ses in sessions:
        session_dict = load_session(subject, ses, split=split)
        ep = preprocess_session(session_dict,low=low,
            high=high,baseline=baseline,threshold_uv=threshold_uv)
        session_epochs.append(ep)

    concatenated = mne.concatenate_epochs(session_epochs, verbose=False)
    logger.info("Preprocessed subject %s (%s, %d sessions) — total %d epochs",
        subject, split, len(sessions), len(concatenated))
    return concatenated
