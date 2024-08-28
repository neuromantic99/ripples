import importlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.pyplot as plt

import mat73
import numpy as np
import pandas as pd
from scipy import io
from ripples.consts import HERE, SAMPLING_RATE_LFP
from ripples.gsheets_importer import gsheet2df
from ripples.models import (
    ClusterInfo,
    ClusterType,
    ProbeCoordinate,
    RipplesSummary,
    RotaryEncoder,
    Session,
)
from ripples.plotting import plot_channel_depth_profile
from ripples.ripple_detection import (
    count_spikes_around_ripple,
    filter_candidate_ripples,
    get_candidate_ripples,
    get_resting_ripples,
    remove_duplicate_ripples,
)

from ripples.utils import (
    degrees_to_cm,
    interleave_arrays,
    smallest_positive_index,
    unwrap_angles,
)
from ripples.utils_npyx import load_lfp_npyx

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe

UMBRELLA = Path("/Volumes/MarcBusche/Jana/Neuropixels")


def preprocess(lfp: np.ndarray) -> np.ndarray:
    """Center and decimate, check what preprocessing is necessary"""
    # lfp = signal.decimate(lfp, 2)
    # return lfp - np.mean(lfp, 0)
    return lfp


def map_channels_to_regions(coordinates: ProbeCoordinate, n_channels: int) -> List[str]:
    """Returns a list of length n-microns with the area at each micron of the probe
    (Work in progress)
    """
    import matlab.engine

    # Clone this: github.com/neuromantic99/neuropixels_trajectory_explorer
    # Add npy matlab to the same folder (github.com/kwikteam/npy-matlab)
    # Set the local path to the repo here:
    path_to_npte = Path("/Users/jamesrowland/Code/neuropixels_trajectory_explorer/")

    eng = matlab.engine.start_matlab()
    eng.cd(str(path_to_npte), nargout=0)
    probe_area_labels, probe_area_boundaries = (
        eng.neuropixels_trajectory_explorer_nogui(
            float(coordinates.AP) / 1000,
            float(coordinates.ML) / 1000,
            float(coordinates.AZ),
            float(coordinates.elevation),
            str(path_to_npte / "npy-matlab"),
            str(HERE.parent / r"Allen CCF Mouse Atlas"),
            nargout=2,
        )
    )

    probe_area_labels = probe_area_labels[0]
    probe_area_boundaries = np.array(probe_area_boundaries).squeeze()

    # Channels with depth 0 are the ones nearest the tip.
    # Channel 0 has depth 0 so start at the tip of the probe
    # (github.com/cortex-lab/neuropixels/issues/16#issuecomment-659604278)

    # from www.nature.com/articles/s41598-021-81127-5/figures/1
    # tip_length = 195
    # Not true but adds simplicity when manually aligning to SW
    tip_length = 0
    distance_from_tip = tip_length

    area_channel: List[str] = []
    for _ in range(n_channels):

        channel_position = int(coordinates.depth - distance_from_tip)
        distance_from_tip += 20

        if channel_position < 0:
            area_channel.append("Outside brain")
            continue

        area_idx = smallest_positive_index(
            channel_position / 1000 - probe_area_boundaries
        )
        # probe_area_labels is 1 shorter than probe_area_boundaries as it
        # marks the start and end of each region. Need to subtract 1 if it's in the
        # final region
        if area_idx == len(probe_area_labels):
            area_idx -= 1

        area_channel.append(probe_area_labels[area_idx])

    return area_channel


def load_lfp(metadata_probe: pd.DataFrame) -> np.ndarray:

    lfp_path = metadata_probe["LFP path"].values[0]
    lfp = load_lfp_npyx(UMBRELLA / lfp_path)

    # The long linear channels are interleaved (confirmed by plotting)
    # TODO: SHOULD THIS FLIP BE HERE AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
    return np.flip(np.concatenate((lfp[0::2, :], lfp[1::2, :]), axis=0), axis=0)


def check_channel_order(clusters_info: List[ClusterInfo]) -> None:
    """Makes sure the deinterleaving is required and has worked"""
    prev = None
    for idx in range(384):
        if c := [cluster for cluster in clusters_info if cluster.channel == idx]:
            if prev is not None:
                assert (
                    c[0].depth == prev + 20
                ), "Channels have been deinterleaved incorrectly. Look at load_spikes"

            prev = c[0].depth
        else:
            prev = None


def load_spikes(
    metadata_probe: pd.DataFrame, region_channel: List[str]
) -> List[ClusterInfo]:

    kilsort_path = metadata_probe["Kilosort path"].values[0]

    # TODO: filter noise clusters, not sure if Jana has vetted these
    # keep_idx = cluster_info[cluster_info["KSLabel"].isin(["good", "mua"])].index

    spike_times = np.load(UMBRELLA / kilsort_path / "spike_times.npy")
    spike_clusters = np.load(UMBRELLA / kilsort_path / "spike_clusters.npy")
    cluster_df = pd.read_csv(UMBRELLA / kilsort_path / "cluster_info.tsv", sep="\t")

    sys.path.append(str(UMBRELLA / kilsort_path))
    params = importlib.import_module("params")
    sys.path.remove(str(UMBRELLA / kilsort_path))

    spike_times = spike_times / params.sample_rate

    n_channels = len(region_channel)
    channel_map = np.arange(n_channels)
    channel_map = interleave_arrays(
        channel_map[: n_channels // 2], channel_map[n_channels // 2 :]
    )

    return [
        ClusterInfo(
            spike_times=(spike_times[spike_clusters == row["cluster_id"]])
            .squeeze()
            .tolist(),
            region=region_channel[row["ch"]],
            info=ClusterType(row["KSLabel"]),
            channel=channel_map[row["ch"]],
            depth=row["depth"],
        )
        for _, row in cluster_df.iterrows()
    ]


def get_smoothed_activity_matrix(
    clusters_info: List[ClusterInfo], sigma: float, region: str | None
) -> np.ndarray:
    """Sigma = gaussian smoothing kernal (seconds)"""

    bin_size = 1 / 1000  # Review this
    sigma_bins = int(sigma / bin_size)

    # TODO: probably have this as argument rather than computing it
    duration = np.max(np.hstack([cluster.spike_times for cluster in clusters_info]))

    time_axis = np.arange(0, duration, bin_size)
    smoothed_activity_matrix = []

    for cluster in clusters_info:

        if cluster.info != ClusterType.GOOD:
            continue

        if region is not None and region not in cluster.region:
            continue

        spike_counts, _ = np.histogram(cluster.spike_times, bins=time_axis)

        spike_counts = spike_counts.astype(
            "float64"
        )  # Otherwise the smoothed result is an integer of all zeros

        smoothed = gaussian_filter1d(spike_counts, sigma=sigma_bins)
        smoothed_activity_matrix.append(smoothed)

    return np.array(smoothed_activity_matrix)


def get_distance_matrix(
    clusters_info: List[ClusterInfo], region: str | None
) -> np.ndarray:
    activity_matrix = get_smoothed_activity_matrix(
        clusters_info,
        sigma=1,
        region=region,
    )
    zscored = zscore(activity_matrix, axis=1)
    crosscorr = np.corrcoef(zscored)
    return 1 - crosscorr


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


def map_channels_to_regions_existing_mat_file(
    metadata_probe: pd.DataFrame,
) -> List[str]:
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


def cache_session(session_name: str, recording_name: str, probe: str) -> None:

    metadata = gsheet2df("1HSERPbm-kDhe6X8bgflxvTuK24AfdrZJzbdBy11Hpcg", "sessions", 1)
    metadata_probe = metadata[
        (metadata["Session"] == session_name)
        & (metadata["Recording Name"] == recording_name)
        & (metadata["Probe"] == probe)
    ]
    coordinates = ProbeCoordinate(
        AP=metadata_probe["AP"],
        ML=metadata_probe["ML"],
        AZ=metadata_probe["AZ"],
        elevation=metadata_probe["Elevation"],
        # depth=metadata_probe["Depth"],
        # TODO: Add a check if this exists
        depth=metadata_probe["Actual depth"],
    )

    lfp = load_lfp(metadata_probe)
    n_channels = lfp.shape[0]
    region_channel = map_channels_to_regions(coordinates, n_channels)
    clusters_info = load_spikes(metadata_probe, region_channel)
    check_channel_order(clusters_info)
    rotary_encoder = load_rotary_encoder(metadata_probe)

    plot_channel_depth_profile(lfp, region_channel, clusters_info)

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

    ripples_summary: Dict[str, Any] = {
        "resting_percentage": num_resting / num_resting_and_running,
    }

    for area in ["retrosplenial", "dentate", "ca1"]:

        channels_keep = [
            idx
            for idx, region in enumerate(region_channel)
            if region is not None and area in region.lower()
        ]

        spike_times = np.hstack(
            [
                cluster.spike_times
                for cluster in clusters_info
                if cluster.channel in channels_keep
            ]
        )

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

        ripples_summary[area] = spike_count

    ripples_summary["ripple_power"] = [ripple.peak_power for ripple in ripples]

    session: Session = Session(
        ripples_summary=RipplesSummary(**ripples_summary),
        clusters_info=clusters_info,
    )

    with open(
        HERE.parent / "results" / f"{session_name}-{recording_name}-Probe{probe}.json",
        "w",
    ) as f:
        json.dump(session.model_dump(), f)


def main() -> None:

    recordings = [
        ("NLGF_A_1393311_3M", "baseline4"),
        ("NLGF_A_1393315_3M", "baseline1"),
        ("WT_A_1397747_3M", "baseline1"),
        ("WT_A_1423496_4M", "baseline1"),
        # (# "WT_A_1412719_6M", "baseline1"),
        # (# "WT_A_1397747_6M", "baseline1"),
        ("NLGF_A_1393314_3M", "baseline1"),
        ("NLGF_A_1393317_3M", "baseline1"),
    ]
    probe = "1"
    for session, recording_name in recordings:
        cache_session(session, recording_name, probe)
