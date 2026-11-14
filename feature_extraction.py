"""
Feature extraction for the P300 xDAWN+LDA pipeline.

Implements spatial filtering with xDAWN and downstream feature
construction for the LDA classifier.
"""

import logging
import numpy as np
import mne
from mne.preprocessing import Xdawn

from config import XDAWN_N_COMPONENTS, SFREQ

logger = logging.getLogger(__name__)


def fit_xdawn(epochs, n_components=XDAWN_N_COMPONENTS, correct_overlap=False):
    #Fit an xDAWN spatial filter on training epochs.

    xdawn = Xdawn(n_components=n_components, correct_overlap=correct_overlap)
    xdawn.fit(epochs)
    logger.info("Fitted Xdawn with n_components=%d", n_components)
    return xdawn


def apply_xdawn(xdawn, epochs):
    """Apply a fitted xDAWN filter to epochs.
    Transforms epochs into the xDAWN spatial component subspace.
    """
    transformed = xdawn.transform(epochs)
    logger.debug("Transformed epochs with xDAWN to shape %s", transformed.shape)
    return transformed


def extract_features(xdawn_epochs):
    #Convert xDAWN-filtered 3-D epochs into a 2-D feature matrix.
    n_epochs = xdawn_epochs.shape[0]
    X = xdawn_epochs.reshape(n_epochs, -1)
    return X


def build_feature_matrix(epochs, targets=None, xdawn=None, n_components=XDAWN_N_COMPONENTS):
    """
    If xdawn is None, fits a new xDAWN on the provided epochs (training mode).
    Otherwise, applies the existing xdawn filter (inference / testing mode).
    """
    if xdawn is None:
        xdawn = fit_xdawn(epochs, n_components=n_components)

    xdawn_epochs = apply_xdawn(xdawn, epochs)
    X = extract_features(xdawn_epochs)

    if targets is None:
        y = epochs.events[:, -1].copy()
    else:
        y = np.asarray(targets)

    return X, y, xdawn
