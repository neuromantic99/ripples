import random
from typing import List

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


from ripples.models import RotaryEncoder, SpikesSession
from ripples.utils import CandidateEvent


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


def plot_spikes_per_region(
    spikes_session: SpikesSession,
    region_channel: List[str],
    plot_all_channels: bool = False,
) -> None:

    if plot_all_channels:
        plt.bar(
            np.arange(0, len(spikes_session.spike_channels)),
            spikes_session.spike_channels,
        )
        plt.xticks(
            np.arange(0, len(spikes_session.spike_channels)),
            region_channel,
            rotation=90,
        )
        return

    total_spikes_per_region = {}

    for idx, region in enumerate(region_channel):
        if region not in total_spikes_per_region:
            total_spikes_per_region[region] = 0
        total_spikes_per_region[region] += spikes_session.spike_channels[idx]

    # take the mean over the number of channels in each region
    total_spikes_per_region = {
        region: int(total_spikes_per_region[region] / region_channel.count(region))
        for region in total_spikes_per_region
    }

    plt.bar(
        np.arange(0, len(total_spikes_per_region)),
        list(total_spikes_per_region.values()),
    )
    plt.xticks(
        np.arange(0, len(total_spikes_per_region)),
        list(total_spikes_per_region.keys()),
        rotation=90,
    )
    # plt.tight_layout()


def plot_ripples_against_position(
    ripples: List[CandidateEvent], rotary_encoder: RotaryEncoder, sampling_rate_lfp: int
) -> None:
    plt.plot(rotary_encoder.time, rotary_encoder.position)
    for ripple in ripples:
        plt.axvline(ripple.onset / sampling_rate_lfp, color="red")
