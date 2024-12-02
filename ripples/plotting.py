import os
import random
from typing import Any, Dict, List

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import zscore
import seaborn as sns


from ripples.consts import SAMPLING_RATE_LFP, HERE
from ripples.models import ClusterInfo, ClusterType, RotaryEncoder, CandidateEvent
from ripples.utils import bandpass_filter, compute_power, moving_average


def plot_ripples(ripples: List[List[CandidateEvent]], filtered_lfp: np.ndarray) -> None:

    assert len(ripples) == filtered_lfp.shape[0]

    length_plot = 500
    idx = 0

    plt.figure(figsize=(10, 10))
    for channel_idx, channel in enumerate(ripples):
        for event in channel:
            if idx == 16:
                break

            ripple_length = event.offset - event.onset
            padding = int((length_plot - ripple_length) / 2)

            ripple = filtered_lfp[
                channel_idx,
                event.onset - padding : event.offset + padding,
            ]

            plt.subplot(4, 4, idx + 1)
            plt.plot(ripple)
            plt.axvline(padding, color="red")
            plt.axvline(event.offset - event.onset + padding, color="red")

            idx += 1


def plot_lfp(lfp: np.ndarray, region_channel: List[str]) -> None:
    already_labelled = {}
    color_idx = -1
    colors = cm.tab20(np.linspace(0, 1, len(set(region_channel))))

    for idx, lfp_channel in enumerate(lfp):

        region = region_channel[idx]

        label_seen = region in already_labelled
        if not label_seen:
            color_idx = random.randint(0, len(colors))

        plt.plot(
            lfp_channel[::1000] + idx * 500,
            label=(None if label_seen else "None" if region is None else region),
            color=colors[color_idx % len(colors)],
            # color="red" if region is not None and "CA1" in region else "black",
        )
        already_labelled[region] = True

    plt.legend()
    plt.show()


def plot_ripples_against_position(
    ripples: List[CandidateEvent], rotary_encoder: RotaryEncoder, sampling_rate_lfp: int
) -> None:
    plt.plot(rotary_encoder.time, rotary_encoder.position)
    for ripple in ripples:
        plt.axvline(ripple.onset / sampling_rate_lfp, color="red")


def plot_frequency_depth(lfp: np.ndarray, ax: Any | None = None) -> None:
    lfp = lfp[:, : SAMPLING_RATE_LFP * 360]
    swr_power = compute_power(
        bandpass_filter(lfp, 125, 250, SAMPLING_RATE_LFP, order=4)
    )
    # 4th order doesn't work at these frequencies
    theta_power = compute_power(bandpass_filter(lfp, 4, 8, SAMPLING_RATE_LFP, order=3))
    delta_power = compute_power(bandpass_filter(lfp, 1, 3, SAMPLING_RATE_LFP, order=3))
    plotting_class = ax if ax is not None else plt
    plotting_class.plot(zscore(swr_power,nan_policy='omit'), label="SWR")
    plotting_class.plot(zscore(theta_power,nan_policy='omit'), label="Theta")
    plotting_class.plot(zscore(delta_power,nan_policy='omit'), label="Delta")

    if ax is None:
        plt.ylabel("Power (z-scored)")
        plt.xlabel("Channel")
    else:
        ax.set_ylabel("Power (z-scored)")
        ax.set_xlabel("Channel")


def plot_channel_depth_profile(
    lfp: np.ndarray,
    region_channel: List[str],
    clusters_info: List[ClusterInfo],
    recording_id: str,
) -> None:

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plot_frequency_depth(lfp, ax1)

    n_spikes_per_channel = [0] * 384
    for cluster in clusters_info:
        if cluster.info in [ClusterType.GOOD, ClusterType.MUA]:
            n_spikes_per_channel[cluster.channel] += len(cluster.spike_times)

    n_spikes_per_channel = moving_average(np.array(n_spikes_per_channel), 3)

    region_channel = np.array(region_channel)

    plot_xlabels: Dict[str, List[float]] = {}
    for region in set(region_channel):

        if region not in ["Outside brain", "CA1"]:
            continue

        region_idxs = np.where(region_channel == region)[0]
        plot_xlabels[region] = [
            np.min(region_idxs).astype(float),
            np.mean(region_idxs).astype(float),
            np.max(region_idxs).astype(float),
        ]

    ax2.plot(n_spikes_per_channel, label="Number of spikes", color="red")
    ax2.set_ylabel("Total number of spikes")
    ax2.set_xticks(
        [position[1] for position in plot_xlabels.values()],
        list(plot_xlabels.keys()),
        rotation=90,
    )
    # ax2.set_xticks(np.arange(384)[::2], np.arange(384)[::2].astype(str), rotation=90)

    for position in plot_xlabels.values():
        plt.axvline(position[0], color="black", linestyle="--")
        plt.axvline(position[2], color="black", linestyle="--")
    ax1.legend()
    ax2.legend(loc="center right")

    figure_path = HERE.parent / "figures" / "depth_profiles"
    if not figure_path.exists():
        os.makedirs(figure_path)

    plt.savefig(figure_path / f"{recording_id}_depth_profile.png")


def plot_lfp_spectrogram(lfp: np.ndarray, recording_id: str) -> None:
    result = []
    lfp = lfp[:, : SAMPLING_RATE_LFP * 180]

    max_freq = 550
    edges = (
        list(range(2, 10, 1))
        + list(range(10, 100, 10))
        + list(range(100, max_freq, 50))
    )

    for idx in range(len(edges) - 1):
        start = edges[idx]
        end = edges[idx + 1]

        result.append(
            compute_power(bandpass_filter(lfp, start, end, SAMPLING_RATE_LFP, order=3))
        )

    result = np.array(result).T
    result[15, :] = 0
    result = np.log(result)
    result[result == -np.inf] = 0
    # result = zscore(result, axis=0)
    sns.heatmap(
        result,
        square=False,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        cbar_kws={"label": "Log power"},
    )
    plt.xticks(range(len(edges)), edges)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Channel")

    figure_path = HERE.parent / "figures" / "lfp_spectrograms"
    if not figure_path.exists():
        os.makedirs(figure_path)

    plt.savefig(figure_path / f"{recording_id}-spectrogram.png")
