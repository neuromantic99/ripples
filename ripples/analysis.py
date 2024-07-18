import importlib
import json
from pathlib import Path


from pathlib import Path
import sys
from typing import Any, Dict, List

import mat73
import numpy as np
import pandas as pd
from scipy import io
from ripples.consts import HERE, SAMPLING_RATE_LFP
from ripples.gsheets_importer import gsheet2df
from ripples.models import RipplesSummary, RotaryEncoder, SpikesSession
from ripples.ripple_detection import (
    count_spikes_around_ripple,
    filter_candidate_ripples,
    get_candidate_ripples,
    get_resting_ripples,
    remove_duplicate_ripples,
)

from ripples.utils import degrees_to_cm, unwrap_angles
from ripples.utils_npyx import load_lfp_npyx

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe

UMBRELLA = Path("/Volumes/MarcBusche/Jana/Neuropixels")


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
    spike_channels = np.array([channel_lookup[cluster] for cluster in spike_clusters])
    return SpikesSession(spike_times=spike_times, spike_channels=spike_channels)


def load_rotary_encoder(metadata_probe: pd.DataFrame) -> RotaryEncoder:
    """scipy io loads this in an insane way"""
    rotary_encoder = io.loadmat(
        UMBRELLA / metadata_probe["Rotary encoder path"].values[0]
    )["data"][0][0]
    positions = rotary_encoder[1][0]
    position_cm = degrees_to_cm(unwrap_angles(positions))
    assert (
        np.max(np.abs(np.diff(position_cm))) < 1
    ), "Something has probably gone wrong with the unwrapping"

    time = rotary_encoder[2][0]
    assert np.all(np.diff(time) >= 0)
    assert positions.shape[0] == time.shape[0]
    return RotaryEncoder(time=time, position=position_cm)


def get_region_channels(metadata_probe: pd.DataFrame) -> List[str]:
    probe_details_path = metadata_probe["Probe Details Path"].values[0]
    try:
        mat_file = mat73.loadmat(UMBRELLA / probe_details_path)
        probe_details = mat_file["probe_details"]
        region_channel = [
            region[0] if region[0] is not None else "None"
            for region in probe_details["alignedregions"]
        ]
    except TypeError:
        mat_file = io.loadmat(UMBRELLA / probe_details_path)
        # Scipy loads this in a cracked out way
        regions = mat_file["probe_details"]["alignedregions"][0][0]
        region_channel = [
            region[0][0] if len(region[0][0]) > 0 else "None" for region in regions
        ]

    # I think the channels are reversed relative to the probe according to Jana's code.
    # Checked with plotting
    return list(reversed(region_channel))


def cache_ripple_result(session: str, recording_name: str, probe: str) -> None:

    metadata = gsheet2df("1HSERPbm-kDhe6X8bgflxvTuK24AfdrZJzbdBy11Hpcg", "Sheet1", 1)
    metadata_probe = metadata[
        (metadata["Session"] == session)
        # & (metadata["Recording Name"] == recording_name)    ## PUT THIS BACK EVENTUALLY
        & (metadata["Probe"] == probe)
    ]

    region_channel = get_region_channels(metadata_probe)
    spike_session = load_spikes(metadata_probe)
    lfp = load_lfp(metadata_probe)

    rotary_encoder = load_rotary_encoder(metadata_probe)

    CA1_channels = [
        idx
        for idx, region in enumerate(region_channel)
        if region is not None and "CA1" in region
    ]

    candidate_events = get_candidate_ripples(
        lfp[CA1_channels, :], sampling_rate=SAMPLING_RATE_LFP
    )

    common_average = np.mean(lfp[CA1_channels, :], axis=0)

    ripples_channels = filter_candidate_ripples(
        candidate_events, lfp[CA1_channels, :], common_average, SAMPLING_RATE_LFP
    )

    # Flattening makes further processing easier but loses the channel information
    ripples = [event for events in ripples_channels for event in events]
    ripples = remove_duplicate_ripples(ripples, 0.3, SAMPLING_RATE_LFP)

    num_resting_and_running = len(ripples)
    print(f"Number of ripples before running removal: {num_resting_and_running}")
    threshold = 1  # Check if this is correct
    ripples = get_resting_ripples(ripples, rotary_encoder, threshold, SAMPLING_RATE_LFP)
    num_resting = len(ripples)
    print(f"Number of ripples after running removal: {num_resting}")

    padding = 2
    n_bins = 200

    result: Dict[str, Any] = {
        "resting_percentage": num_resting / num_resting_and_running
    }

    for area in ["retrosplenial", "dentate", "ca1"]:

        channels_keep = [
            idx
            for idx, region in enumerate(region_channel)
            if region is not None and area in region.lower()
        ]

        spike_times = spike_session.spike_times[
            np.isin(spike_session.spike_channels, channels_keep)
        ]

        spike_count = [
            count_spikes_around_ripple(
                ripple=ripple,
                spike_times=spike_times,
                padding=padding,
                num_bins=n_bins,
                sampling_rate_lfp=SAMPLING_RATE_LFP,
            )
            for ripple in ripples
        ]

        result[area] = spike_count

    result["ripple_power"] = [ripple.peak_power for ripple in ripples]

    RipplesSummary.model_validate(result)

    with open(
        HERE.parent / "results" / f"{session}-{recording_name}-Probe{probe}.json", "w"
    ) as f:
        json.dump(result, f)


def main() -> None:

    sessions = [
        # "NLGF_A_1393311_3M",
        "NLGF_A_1393315_3M",
        "WT_A_1397747_3M",
        "WT_A_1423496_4M",
        # "WT_A_1412719_6M",
        # "WT_A_1397747_6M",
        "NLGF_A_1393314_3M",
        "NLGF_A_1393317_3M",
    ]
    recording_name = "baseline1"
    probe = "1"
    for session in sessions:
        cache_ripple_result(session, recording_name, probe)
