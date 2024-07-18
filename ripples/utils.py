from typing import List, TypeVar

import numpy as np
from scipy import signal


def bandpass_filter(
    lfp: np.ndarray, low: int, high: int, sampling_rate: int
) -> np.ndarray:
    b, a = signal.butter(4, Wn=[low, high], fs=sampling_rate, btype="bandpass")
    return signal.filtfilt(b, a, lfp, axis=1)


def compute_envelope(lfp: np.ndarray) -> np.ndarray:
    hilbert_transformed = signal.hilbert(lfp, axis=1)
    return np.abs(hilbert_transformed)


def shuffle(x: np.ndarray) -> np.ndarray:
    """shuffles along all dimensions of an array"""
    shape = x.shape
    x = np.ravel(x)
    np.random.shuffle(x)
    return x.reshape(shape)


def flatten(xss: List[List]) -> List:
    return [x for xs in xss for x in xs]


T = TypeVar("T", float, np.ndarray)


def degrees_to_cm(degrees: T) -> T:
    WHEEL_CIRUMFERENCE = 48  # Have not actually measured the ephys rig.
    return (degrees / 360) * WHEEL_CIRUMFERENCE


def forward_fill(arr: np.ndarray) -> np.ndarray:
    """numpy single dimension array equivalent of pandas ffill"""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.size), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def smallest_positive_index(arr: np.ndarray) -> int:
    return np.where(arr >= 0, arr, np.inf).argmin().astype(int)


def unwrap_angles(angles: np.ndarray) -> np.ndarray:
    return np.unwrap(np.deg2rad(angles), period=np.pi) * (180 / np.pi)
