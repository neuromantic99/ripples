from inspect import iscode
from pathlib import Path
from typing import Iterable, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
import mat73

from ripples.consts import HERE, 2500
from ripples.utils import bandpass_filter, compute_envelope, compute_power
from ripples.utils_npyx import load_lfp_npyx


def cache_lfp() -> None:
    UMBRELLA = Path("/Volumes/MarcBusche/Qichen/Neuropixels/")
    # folder = Path("ELGH11144/20240808_g0/20240808_g0_imec0")
    folder = Path("ELGH11144/20240808_g1/20240808_g1_imec0")
    print("LOADING")
    lfp = load_lfp_npyx(str(UMBRELLA / folder))
    print("DONE LOADING")
    np.save(HERE.parent / "data" / "lfp_sleep_long.npy", lfp)


def plot_frequencies_over_time(lfp: np.ndarray, channels: Iterable[int]) -> None:

    lfp = lfp[:, int(2500 * 44.7) :]

    window_size = 3 * 2500
    step = int(0.1 * 2500)
    theta = []
    delta = []
    slow_wave = []
    high_gamma = []
    ratio = []
    sigma = []
    n_samples = lfp.shape[1]

    # for window_start in range(0, n_samples - window_size, window_size):
    for window_start in range(0, n_samples - window_size, step):

        print(f"data from {window_start} to {window_start + window_size}")

        data = lfp[channels, window_start : window_start + window_size]

        # theta.append(
        #     compute_envelope(bandpass_filter(data, 4, 8, 2500)).mean()
        # )
        # delta.append(
        #     compute_envelope(bandpass_filter(data, 1, 3, 2500)).mean()
        # )
        slow_wave.append(
            compute_envelope(bandpass_filter(data, 0.5, 4, 2500)).mean()
        )
        # high_gamma.append(
        #     compute_envelope(bandpass_filter(data, 100, 250, 2500)).mean()
        # )

        # sigma.append(
        #     compute_envelope(bandpass_filter(data, 10, 15, 2500)).mean()
        # )

    # plt.plot(zscore(np.array(theta) / np.array(delta)), label="Theta / Delta")
    # plt.plot(zscore(delta), label="Delta")
    # plt.plot(norm(moving_average(np.array(theta), 5)), label="Theta")
    # x_axis = np.arange(
    #     0, len(theta) * window_size / 2500, window_size / 2500
    # )
    slow_wave = zscore(np.array(slow_wave))
    # plt.plot(x_axis, zscore(np.array(high_gamma)), label="high gamma")
    # plt.plot(x_axis, slow_wave, label="slow wave")
    # plt.plot(x_axis, zscore(np.array(sigma)), label="Sigma")
    np.save(HERE.parent / "data" / "slow_wave.npy", slow_wave)
    # np.save(HERE.parent / "data" / "slow_wave_xaxis.npy", x_axis)

    # plt.legend()

    # plt.xlabel("Time (s)")
    # plt.savefig(HERE.parent / "figures" / "slow wave.png")
    # plt.show()


def load_sync() -> np.ndarray:
    sync = mat73.loadmat(HERE.parent / "data" / "20240808_g0_t0.sync.mat")["sync"]
    1 / 0


def main() -> None:
    # cache_lfp()
    # load_sync()
    lfp = np.load(HERE.parent / "data" / "lfp_sleep.npy")

    # I think this is the reference, double check
    lfp[191, :] = 0
    lfp = np.flip(lfp, axis=0)

    channels = range(210, 300)  # Roughly
    # channels = range(0, 384)
    plot_frequencies_over_time(lfp, channels)

    1 / 0


# bandpass_filter
# n_channels, n_samples = lfp.shape
# theta_band = np.zeros(n_channels)
# delta_band = np.zeros_like(lfp)

# window = 1 * 2500

# for channel in range(n_channels):

#     for w
#     theta_band[channel] = bandpass_filter(lfp[channel, :], 4, 8, 2500)
#     delta_band[channel, :] = bandpass_filter(
#         lfp[channel, :], 1, 4, 2500
#     )
