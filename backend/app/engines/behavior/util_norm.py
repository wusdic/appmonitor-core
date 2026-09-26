"""Normalisation helper shared by fingerprint & clustering engines."""
from __future__ import annotations

import numpy as np


def zscore_normalise(mat: np.ndarray) -> np.ndarray:
    """Column-wise z-normalisation; constant columns collapse to 0."""
    if mat.size == 0:
        return mat
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    return (mat - mean) / std
