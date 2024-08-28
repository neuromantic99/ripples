import random
from typing import Any, Dict, List

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import zscore


from ripples.consts import SAMPLING_RATE_LFP
from ripples.models import ClusterInfo, RotaryEncoder, CandidateEvent
from ripples.utils import bandpass_filter, compute_power


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
    plotting_class.plot(zscore(swr_power), label="SWR")
    plotting_class.plot(zscore(theta_power), label="Theta")
    plotting_class.plot(zscore(delta_power), label="Delta")

    if ax is None:
        plt.ylabel("Power (z-scored)")
        plt.xlabel("Channel")
    else:
        ax.set_ylabel("Power (z-scored)")
        ax.set_xlabel("Channel")


def plot_channel_depth_profile(
    lfp: np.ndarray, region_channel: List[str], clusters_info: List[ClusterInfo]
) -> None:

    # depth_map: Dict[float, float] = {}

    # for cluster in clusters_info:
    #     if cluster.depth not in depth_map:
    #         depth_map[cluster.depth] = 0
    #     depth_map[cluster.depth] += len(cluster.spike_times)

    # plt.plot(depth_map.keys(), depth_map.values(), ".")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    plot_frequency_depth(lfp, ax1)

    n_spikes_per_channel = [0] * 384
    for cluster in clusters_info:
        # if cluster.info == ClusterType.GOOD:
        n_spikes_per_channel[cluster.channel] += len(cluster.spike_times)

    region_channel = np.array(region_channel)
    plot_xlabels: Dict[str, List[float]] = {}
    for region in set(region_channel):
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
    # ax2.set_xticks(
    #     np.arange(384)[::2], (np.arange(384) * 20)[::2].astype(str), rotation=90
    # )

    for position in plot_xlabels.values():
        plt.axvline(position[0], color="black", linestyle="--")
    ax1.legend()
    ax2.legend()
    plt.show()
