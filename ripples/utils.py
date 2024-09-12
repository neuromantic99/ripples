from collections import defaultdict
from typing import List, TypeVar

import numpy as np
from scipy import signal

from ripples.models import SessionToAverage


def bandpass_filter(
    lfp: np.ndarray, low: float, high: float, sampling_rate: int, order: int = 4
) -> np.ndarray:
    b, a = signal.butter(order, Wn=[low, high], fs=sampling_rate, btype="bandpass")
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


def compute_power(filtered_data: np.ndarray) -> np.ndarray:
    """
    Compute the average power of the filtered data over each window.

    Parameters:
    - filtered_data: ndarray (n_channels, window_size_samples), filtered signal

    Returns:
    - power: ndarray (n_channels,), average power in each channel

    TODO: IS THIS CORRECT?
    """
    return np.mean(filtered_data**2, axis=1)


def moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    return np.convolve(arr, np.ones(window), "valid") / window


def norm(x: np.ndarray) -> np.ndarray:
    return (x - min(x)) / (max(x) - min(x))


def threshold_detect(signal: np.ndarray, threshold: float) -> np.ndarray:
    """Returns the indices where signal crosses the threshold"""
    thresh_signal = signal > threshold
    thresh_signal[1:][thresh_signal[:-1] & thresh_signal[1:]] = False
    times = np.where(thresh_signal)
    return times[0]


def interleave_arrays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c = np.empty((a.size + b.size,), dtype=a.dtype)
    c[::2] = a
    c[1::2] = b
    return c


def mean_across_same_session(sessions: List[SessionToAverage]) -> List[float]:

    data_dict: defaultdict[str, dict] = defaultdict(lambda: {"sum": 0, "count": 0})
    # Populate the dictionary with sums and counts
    for session in sessions:
        data_dict[session.id]["sum"] += session.data
        data_dict[session.id]["count"] += 1

    return [value["sum"] / value["count"] for value in data_dict.values()]
