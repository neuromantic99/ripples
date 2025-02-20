from collections import defaultdict
from typing import List, TypeVar

import numpy as np
from scipy import signal, trapz
from matplotlib import pyplot as plt
from scipy import signal

from ripples.models import SessionToAverage


def bandpass_filter(
    lfp: np.ndarray, low: float, high: float, sampling_rate: float, order: int = 4
) -> np.ndarray:
    b, a = signal.butter(order, Wn=[low, high], fs=sampling_rate, btype="bandpass")
    return signal.filtfilt(b, a, lfp, axis=1)


# adapted from https://www.askpython.com/python-modules/pandas/comparing-bandpower-matlab-python-numpy
# still needs adjustments
def bandpower(
    lfp: np.ndarray, sampling_rate: float, fmin: int, fmax: int, method: str
) -> float:
    if method == "welch":
        freqrange = [fmin, fmax]
        frequencies, psd = signal.welch(
            lfp, sampling_rate, nperseg=sampling_rate, scaling="density"
        )
        freq_indices = np.where(
            (frequencies >= freqrange[0]) & (frequencies <= freqrange[1])
        )
        band_power = np.trapz(
            psd.reshape(np.size(psd))[freq_indices], frequencies[freq_indices]
        )
    elif method == "periodogram":
        f, Pxx = signal.periodogram(lfp, fs=sampling_rate)
        ind_min = np.argmax(f > fmin) - 1
        ind_max = np.argmax(f > fmax) - 1
        band_power = trapz(Pxx[ind_min:ind_max], f[ind_min:ind_max])

    return band_power


def get_event_frequency(
    lfp: np.ndarray, sampling_rate: float, plot: bool = False
) -> float:
    [f, Pxx] = signal.periodogram(lfp, fs=sampling_rate)
    max_idx = np.argmax(Pxx.reshape(len(f), 1).tolist())
    max_freq = f[max_idx]
    max_val = (Pxx.reshape(len(f), 1)).tolist()[max_idx]
    max_val = max_val[0]
    peaks, _ = signal.find_peaks(Pxx, height=0.25 * max_val)
    if list(peaks):
        peaks_freq = np.array(f[peaks])
        if sum(peaks_freq > 100) == 1:
            ev_freq = peaks_freq[peaks_freq > 100]
        elif sum(peaks_freq > 100) == 0:
            ev_freq = peaks_freq[np.argmax(Pxx[peaks])]
        elif sum(peaks_freq > 100) > 1:
            Pxx = np.array(Pxx)
            ev_freq = peaks_freq[Pxx[peaks] == max(Pxx[peaks[peaks_freq > 100]])]

        return float(ev_freq)
    else:
        print(
            "Ripple event with frequency peak at the border of the spectrum - will be excluded"
        )
        return 1


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
