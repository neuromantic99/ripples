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
