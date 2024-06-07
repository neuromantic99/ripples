import importlib
from pathlib import Path


from collections import Counter
from pathlib import Path
import sys

import mat73
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ripples.gsheets_importer import gsheet2df
from ripples.models import SpikesSession
from ripples.plotting import plot_ripples
from ripples.ripple_detection import filter_candidate_ripples
from ripples.utils import bandpass_filter, get_candidate_ripples
from ripples.utils_npyx import load_lfp_npyx

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe


UMBRELLA = Path("/Users/jamesrowland/Documents/data/")
SESSION = "NLGF_A_1393311_3M"
RECORDING_NAME = "baseline1"
PROBE = "1"


SAMPLING_RATE = 2500


def preprocess(lfp: np.ndarray) -> np.ndarray:
    """Center and decimate, check what preprocessing is necessary"""
    # lfp = signal.decimate(lfp, 2)
    # return lfp - np.mean(lfp, 0)
    return lfp


def load_lfp(metadata_probe: pd.DataFrame) -> np.ndarray:

    lfp_path = metadata_probe["LFP path"].values[0]
    lfp = load_lfp_npyx(UMBRELLA / lfp_path)

    # The long linear channels are interleaved (confirmed by plotting)
    return np.concatenate((lfp[0::2, :], lfp[1::2, :]), axis=0)


def load_spikes(metadata_probe: pd.DataFrame) -> SpikesSession:

    kilsort_path = metadata_probe["Kilosort path"].values[0]

    # TODO: filter noise clusters, not sure if Jana has vetted these
    # keep_idx = cluster_info[cluster_info["KSLabel"].isin(["good", "mua"])].index

    spike_times = np.load(UMBRELLA / kilsort_path / "spike_times.npy")
    spike_clusters = np.load(UMBRELLA / kilsort_path / "spike_clusters.npy")
    cluster_info = pd.read_csv(UMBRELLA / kilsort_path / "cluster_info.tsv", sep="\t")

    sys.path.append(str(UMBRELLA / kilsort_path))
    params = importlib.import_module("params")
    sys.path.remove(str(UMBRELLA / kilsort_path))

    spike_times = spike_times / params.sample_rate

    channel_lookup = dict(zip(cluster_info["cluster_id"], cluster_info["ch"]))
    spike_channels_map = Counter(
        [channel_lookup[cluster] for cluster in spike_clusters]
    )
    spike_channels_map.update(
        {idx: 0 for idx in range(384) if idx not in spike_channels_map}
    )
    spike_channels = np.array(
        [spike_channels_map[idx] for idx in sorted(spike_channels_map.keys())]
    )
    spike_channels = np.concatenate((spike_channels[::2], spike_channels[1::2]), axis=0)

    return SpikesSession(spike_times=spike_times, spike_channels=spike_channels)


def get_region_channels(metadata_probe: pd.DataFrame):
    probe_details_path = metadata_probe["Probe Details Path"].values[0]
    probe_details = mat73.loadmat(UMBRELLA / probe_details_path)["probe_details"]

    region_channel = [
        region[0] if region[0] is not None else "None"
        for region in probe_details["alignedregions"]
    ]
    # I think the channels are reversed relative to the probe according to Jana's code.
    # Checked with plotting
    return list(reversed(region_channel))


def main():

    metadata = gsheet2df("1HSERPbm-kDhe6X8bgflxvTuK24AfdrZJzbdBy11Hpcg", "Sheet1", 1)
    metadata_probe = metadata[
        (metadata["Session"] == SESSION)
        & (metadata["Recording Name"] == RECORDING_NAME)
        & (metadata["Probe"] == PROBE)
    ]

    region_channel = get_region_channels(metadata_probe)
    spike_session = load_spikes(metadata_probe)
    lfp = load_lfp(metadata_probe)

    CA1_channels = [
        idx
        for idx, region in enumerate(region_channel)
        if region is not None and "CA1" in region
    ]

    # plot_lfp(lfp, region_channel)
    # plot_spikes_per_region(spike_session, region_channel, plot_all_channels=True)

    candidate_events = get_candidate_ripples(
        lfp[CA1_channels, :], sampling_rate=SAMPLING_RATE
    )

    common_average = np.mean(lfp[CA1_channels, :], axis=0)

    ripples = filter_candidate_ripples(
        candidate_events, lfp[CA1_channels, :], common_average, SAMPLING_RATE
    )

    num_events = sum(len(events) for events in ripples)
    print(f"Number of events: {num_events}")

    filtered = bandpass_filter(lfp[CA1_channels, :], 125, 250, SAMPLING_RATE)
    plot_ripples(ripples, filtered)
    plt.show()
