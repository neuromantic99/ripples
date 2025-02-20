import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore
import matplotlib.pyplot as plt
import csv

import mat73
import numpy as np
import pandas as pd
from scipy import io
from ripples.consts import HERE
from ripples.gsheets_importer import gsheet2df
from ripples.models import (
    ClusterInfo,
    ClusterType,
    ProbeCoordinate,
    RipplesSummary,
    RotaryEncoder,
    Session,
)
from ripples.plotting import (
    plot_channel_depth_profile,
    plot_lfp_spectrogram,
    plot_resting_ripples,
)
from ripples.ripple_detection import (
    count_spikes_around_ripple,
    filter_candidate_ripples,
    get_candidate_ripples,
    remove_duplicate_ripples,
    get_quality_metrics,
)

from ripples.utils import (
    bandpass_filter,
    compute_power,
    degrees_to_cm,
    interleave_arrays,
    smallest_positive_index,
    unwrap_angles,
    threshold_detect,
)
from ripples.utils_npyx import load_lfp_npyx

from ripples.consts import RIPPLE_BAND

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe

UMBRELLA = Path("//128.40.224.64/marcbusche/Jana/Neuropixels")


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
    path_to_npte = Path("C:/Python_code/neuropixels_trajectory_explorer")

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


def load_lfp(lfp_path: Path) -> Tuple[np.ndarray, np.ndarray, float]:

    lfp, sync, sampling_rate_lfp = load_lfp_npyx(str(UMBRELLA / lfp_path))

    # The long linear channels are interleaved (confirmed by plotting)
    lfp = np.concatenate((lfp[0::2, :], lfp[1::2, :]), axis=0)

    assert len(sync) == lfp.shape[1] 

    # Chop off beginning & end of the recording without behavioural data
    rising_edges = threshold_detect(sync, 0.5)
    assert rising_edges[4] - rising_edges[0] == 0.8*sampling_rate_lfp 
    assert rising_edges[-1] - rising_edges[-5] == 0.8*sampling_rate_lfp
    # sync signal consists in one 1s 5Hz pulse at the end and at the beginning of the recording, in between there is a 1Hz pulse
    # behavioural recording starts at the first rising edge of the 1s 5Hz pulse at the beginning of the recording
    recording_onset = rising_edges[0]
    # behavioural recording stops at the first rising edge of the 1s 5 Hz pulse, sampling rate 2500 Hz
    recording_offset = rising_edges[-5]
    lfp_chopped = lfp[:, recording_onset:recording_offset]

    assert lfp_chopped.shape[1] > 300 * sampling_rate_lfp  # longer than 5mins

    return lfp_chopped, sync, sampling_rate_lfp


def lfp_clear_internal_reference_channel(lfp: np.ndarray) -> np.ndarray:
    int_ref_channel = 191
    lfp = lfp.astype(float)
    lfp[int_ref_channel, :] = np.nan
    return lfp


def lfp_get_noise_levels(lfp: np.ndarray) -> List[float]:
    rms_per_channel = np.sqrt(np.nanmean(lfp**2, axis=1))
    rms_per_channel = rms_per_channel.tolist()
    return rms_per_channel


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
    kilosort_path: Path,
    region_channel: List[str],
    sync: np.ndarray,
    sampling_rate_lfp: float,
    plot: bool = False,
) -> List[ClusterInfo]:

    # TODO: filter noise clusters, not sure if Jana has vetted these
    # keep_idx = cluster_info[cluster_info["KSLabel"].isin(["good", "mua"])].index

    spike_times = np.load(UMBRELLA / kilosort_path / "spike_times.npy")
    spike_clusters = np.load(UMBRELLA / kilosort_path / "spike_clusters.npy")
    cluster_df = pd.read_csv(UMBRELLA / kilosort_path / "cluster_info.tsv", sep="\t")

    assert len(spike_times) == len(spike_clusters)
    spike_cluster_list = list(set(spike_clusters))
    cluster_id_list = list(set(cluster_df["cluster_id"]))
    assert spike_cluster_list == cluster_id_list

    sys.path.append(str(UMBRELLA / kilosort_path))
    params = importlib.import_module("params")
    sys.path.remove(str(UMBRELLA / kilosort_path))
    sampling_rate_spikes = params.sample_rate

    # Chop off beginning & end of the recording without behavioural data
    rising_edges = threshold_detect(sync, 0.5)
    recording_onset = rising_edges[0] / sampling_rate_lfp * sampling_rate_spikes
    # behavioural recording stops at the first rising edge of the 1s 5 Hz pulse, sampling rate 2500 Hz
    recording_offset = rising_edges[-5] / sampling_rate_lfp * sampling_rate_spikes

    aligned_spike_times_ind = (spike_times > recording_onset) & (
        spike_times < recording_offset
    )
    aligned_spike_times = (
        spike_times[aligned_spike_times_ind] - recording_onset
    )  # chop off beginning & end, set the time so that it alignes to rotary encoder
    aligned_spike_times = aligned_spike_times / sampling_rate_spikes
    aligned_spike_clusters = spike_clusters.reshape(len(spike_clusters), 1)[
        aligned_spike_times_ind
    ]

    assert (
        min(aligned_spike_times) < 1
    )  # to check if chopping works, assuming that there is at least one spike in the first second of the recording
    assert len(aligned_spike_times) == len(aligned_spike_clusters)

    if plot:
        # bin the data to get an idea of spiking activity over the recording
        num_bins = int(max(np.round(aligned_spike_times)))
        # Use numpy's histogram function for equal width bins
        hist, _ = np.histogram(aligned_spike_times, bins=num_bins)
        plt.figure()
        plt.hist(hist, bins="auto")

        # Scatterplot, very slow, need to find a way to improve this if needed
        spike_depths = [
            cluster_df["depth"][int(np.where(cluster_df["cluster_id"] == cluster)[0])]
            for cluster in spike_clusters
        ]
        aligned_spike_depths = np.array(spike_depths).reshape(len(spike_depths), 1)[
            aligned_spike_times_ind
        ]
        plt.figure()
        plt.scatter(aligned_spike_times, aligned_spike_depths)
        plt.show()

    n_channels = len(region_channel)
    channel_map = np.arange(n_channels)
    channel_map = interleave_arrays(
        channel_map[: n_channels // 2], channel_map[n_channels // 2 :]
    )

    return [
        ClusterInfo(
            spike_times=(
                aligned_spike_times[aligned_spike_clusters == row["cluster_id"]]
            )
            .squeeze()
            .tolist(),
            region=region_channel[row["ch"]],
            info=ClusterType(row["KSLabel"]),
            channel=int(
                channel_map[row["ch"]]
            ),  # Int cast as np.in64 is not json serializable
            depth=row["depth"],
        )
        for _, row in cluster_df.iterrows()
        # A single spike doesn't make sense and it ruins the type of the class
        if len(aligned_spike_times[aligned_spike_clusters == row["cluster_id"]]) > 1
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


def load_rotary_encoder(rotary_encoder_path: Path) -> RotaryEncoder:
    """scipy io loads this in an insane way"""
    rotary_encoder = io.loadmat(UMBRELLA / rotary_encoder_path)["data"][0][0]
    positions = rotary_encoder[1][0]
    position_cm = degrees_to_cm(unwrap_angles(positions))
    assert (
        np.max(np.abs(np.diff(position_cm))) < 1
    ), "Something has probably gone wrong with the unwrapping"

    time = rotary_encoder[2][0]

    assert np.all(np.diff(time) >= 0)
    assert positions.shape[0] == time.shape[0]
    return RotaryEncoder(time=time, position=position_cm)


def calculate_speed(
    idx: int,
    bin_edges_ind: np.ndarray,
    rotary_encoder: RotaryEncoder,
    sampling_rate: float,
) -> np.ndarray:
    start_time = bin_edges_ind[idx] / sampling_rate
    end_time = bin_edges_ind[idx + 1] / sampling_rate
    start_idx = smallest_positive_index(start_time - rotary_encoder.time)
    end_idx = smallest_positive_index(end_time - rotary_encoder.time)
    distance = rotary_encoder.position[end_idx] - rotary_encoder.position[start_idx]
    speed_cm_per_s = distance / (end_time - start_time)
    return speed_cm_per_s


def get_resting_periods(
    rotary_encoder: RotaryEncoder, sampling_rate: float, max_time: float
) -> Tuple[np.ndarray, np.ndarray]:

    bin_size = sampling_rate
    bin_edges_ind = np.arange(0, max_time, bin_size)
    speed_cm_per_s = np.array([])

    for idx in range(len(bin_edges_ind) - 1):
        speed_bin = calculate_speed(idx, bin_edges_ind, rotary_encoder, sampling_rate)
        speed_cm_per_s = np.concatenate(
            (speed_cm_per_s, np.full(int(round(sampling_rate)), speed_bin))
        )


    last_idx = int(bin_edges_ind[-1])
    max_time = max_time
    if max_time > last_idx:
        start_time = last_idx / sampling_rate
        end_time = max_time / sampling_rate
        start_idx = smallest_positive_index(start_time - np.array(rotary_encoder.time))
        end_idx = smallest_positive_index(end_time - np.array(rotary_encoder.time))
        distance = rotary_encoder.position[end_idx] - rotary_encoder.position[start_idx]
        speed_extra_time = distance / (end_time - start_time)
        speed_cm_per_s = np.concatenate(
            (speed_cm_per_s, np.full(int(max_time) - last_idx, speed_extra_time))
        )

    assert len(speed_cm_per_s) == max_time

    resting_ind = speed_cm_per_s == 0
    assert len(resting_ind) == max_time

    return resting_ind, speed_cm_per_s


def load_channel_regions(channel_path: Path) -> List[str]:
    with open(
        channel_path,
        "r",
    ) as f:
        reader = csv.reader(f)
        return list(reader)[0]


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


def cache_session(metadata_probe: pd.Series) -> None:

    # load and preprocess LFP
    lfp_raw, sync, sampling_rate_lfp = load_lfp(Path(metadata_probe["LFP path"]))
    assert int(round(sampling_rate_lfp)) >= 2499 & int(round(sampling_rate_lfp)) <= 2501

    lfp = lfp_clear_internal_reference_channel(lfp_raw)
    rms_per_channel = lfp_get_noise_levels(lfp)
    assert lfp.shape[0] == 384
    assert lfp.shape[1] == lfp_raw.shape[1]
    assert np.isnan(lfp[191, 0])  # reference channel, should be set to NaN

    # load rotary encoder file
    rotary_encoder = load_rotary_encoder(Path(metadata_probe["Rotary encoder path"]))
    # identify resting periods based on running speed
    resting_ind, speed_cm_per_s = get_resting_periods(
        rotary_encoder, sampling_rate_lfp, lfp.shape[1]
    )

    resting_percentage = sum(resting_ind) / len(resting_ind)
    resting_time = sum(resting_ind) / sampling_rate_lfp

    # pull out recording specific data from google sheets
    recording_id = f"{metadata_probe['Session']}-{metadata_probe['Recording Name']}-Probe{metadata_probe['Probe']}"

    # map channels to brain regions (based on neuropixel trajectory explorer)

    depth = (
        metadata_probe["Actual depth"]
        if len(metadata_probe["Actual depth"]) > 0
        else metadata_probe["Depth"]
    )
    print(f"DEPTH: {depth}")

    coordinates = ProbeCoordinate(
        AP=metadata_probe["AP"],
        ML=metadata_probe["ML"],
        AZ=metadata_probe["AZ"],
        elevation=metadata_probe["Elevation"],
        depth=depth,
    )

    channel_path = Path(
        "C:/Python_code/ripples/results/New_code_0702/channel_regions/"
        + recording_id
        + ".csv"
    )

    if channel_path.exists():
        region_channel = load_channel_regions(channel_path)
        print("loading channelmap")

    else:
        # #pulls out the corresponding brain region for each channel using the
        # #neuropixels trajectory explorer (matlab engine)

        region_channel = map_channels_to_regions(coordinates, lfp.shape[0])
        print("loading NPX trajectory explorer to pull out channelmap")

    assert len(region_channel) == 384
    assert region_channel[383] == "Outside brain"

    # load kilosort processed data (already preprocessed using matlab code & phy)
    clusters_info = load_spikes(
        Path(metadata_probe["Kilosort path"]), region_channel, sync, sampling_rate_lfp
    )
    check_channel_order(clusters_info)

    # save brain region for each channel in a seperate file
    data_path_channel_regions = (
        HERE.parent / "results" / "test_1902_5Median" / "channel_regions"
    )
    if not data_path_channel_regions.exists():
        os.makedirs(data_path_channel_regions)
    with open(
        data_path_channel_regions / f"{recording_id}.csv",
        "w",
    ) as f:
        write = csv.writer(f)
        write.writerow(region_channel)

    # plot and save lfp spectrogram
    plot_lfp_spectrogram(lfp, recording_id, sampling_rate_lfp)

    all_CA1_channels = [
        idx
        for idx, region in enumerate(region_channel)
        if region is not None and "CA1" in region
    ]

    # Find CA1 channel with highest Ripple power and +/- two channels to detect ripples, then do CAR
    swr_power = compute_power(
        bandpass_filter(lfp, RIPPLE_BAND[0], RIPPLE_BAND[1], sampling_rate_lfp, order=4)
    )
    max_powerChanCA1 = np.nanargmax(swr_power[all_CA1_channels])

    assert max_powerChanCA1 + 3 < len(all_CA1_channels)
    assert max_powerChanCA1 - 2 >= 0

    # CA1_channels are the channels in CA1 used for ripple detection
    detection_channels_ca1 = all_CA1_channels[max_powerChanCA1 - 2 : max_powerChanCA1 + 3]

    # If the reference channel is part of the selected channels for ripple analysis replace with neighbouring channel with the higher ripple power
    if 191 in detection_channels_ca1:
        detection_channels_ca1.remove(191)
        lower_channel = all_CA1_channels[max_powerChanCA1 - 3]
        if (max_powerChanCA1 + 3) > (len(all_CA1_channels) - 1):
            swr_pow_higher_channel = 0
        else:
            higher_channel = all_CA1_channels[max_powerChanCA1 + 3]
            swr_pow_higher_channel = swr_power[higher_channel]
        if swr_power[lower_channel] > swr_pow_higher_channel:
            detection_channels_ca1.append(lower_channel)
        else:
            detection_channels_ca1.append(higher_channel)

    assert region_channel[min(CA1_channels) : (max(CA1_channels) + 1)] == ["CA1"] * 5 

    CA1_channels_swr_pow = list(swr_power[detection_channels_ca1])
    print(f"CA1_channels: {detection_channels_ca1} , power: {CA1_channels_swr_pow}")

    plot_channel_depth_profile(
        lfp, region_channel, clusters_info, recording_id, detection_channels_ca1
    )

    # CAR: take mean across all CA1 channels and then subtract from each channel
    lfp_all_CA1 = lfp[all_CA1_channels, :]
    lfp_detection_chans = lfp[detection_channels_ca1, :]
    common_average = np.nanmean(lfp_all_CA1, axis=0)
    lfp_detection_chans_CAR = np.subtract(lfp_detection_chans, common_average)

    assert lfp_detection_chans_CAR.shape[0] == len(detection_channels_ca1)
    assert lfp_detection_chans_CAR.shape[1] == len(resting_ind)

    candidate_events = get_candidate_ripples(
        lfp_detection_chans_CAR,
        detection_channels_ca1,
        resting_ind,
        sampling_rate=sampling_rate_lfp,
    )

    print(
        f"Number of ripples before filtering: {len([event for events in candidate_events for event in events])}"
    )

    # ripples_channels = filter_candidate_ripples(
    #     candidate_events, lfp_detection_chans_CAR, common_average, SAMPLING_RATE_LFP
    # )

    ripples_channels = candidate_events

    # Flattening makes further processing easier
    ripples = [event for events in ripples_channels for event in events]

    print(f"Number of ripples after filtering: {len(ripples)}")

    ripples = remove_duplicate_ripples(
        ripples, 0.05, sampling_rate_lfp
    )  # James 0.3, Buzaki 0.12, elife 0.05

    print(f"Number of ripples after removing duplicates: {len(ripples)}")

    freq_check, CAR_check, SRP_check, CAR_check_lr, SRP_check_lr, ripples = (
        get_quality_metrics(
            ripples, lfp_detection_chans_CAR, common_average, sampling_rate_lfp
        )
    )

    plot_resting_ripples(
        rotary_encoder,
        lfp.shape[1],
        ripples,
        resting_ind,
        speed_cm_per_s,
        sampling_rate_lfp,
        recording_id,
    )

    ripples_summary: Dict[str, Any] = {
        "resting_time": resting_time,
        "resting_percentage": resting_percentage,
        "events": ripples,
    }

    area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}

    # Identify clusters for each region and extract spikeTimes around each ripple event for each cluster
    # TODO: Probably this is the wrong place to compute this
    padding = 2  # in seconds
    n_bins = 200  # 200 bins = 100 bins/second
    for area in area_map:

        channels_keep = [
            idx
            for idx, region in enumerate(region_channel)
            if region is not None and area in region.lower()
        ]

        print(f"Number of channels in {area_map[area]}: {len(channels_keep)}")

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
                sampling_rate_lfp=sampling_rate_lfp,
            )
            for ripple in ripples
        ]

        ripples_summary[area_map[area]] = spike_count

    ripples_summary["ripple_amplitude"] = [ripple.peak_amplitude for ripple in ripples]
    ripples_summary["ripple_frequency"] = [ripple.frequency for ripple in ripples]
    ripples_summary["ripple_bandpower"] = [
        ripple.bandpower_ripple for ripple in ripples
    ]
    ripples_summary["ripple_freq_check"] = freq_check
    ripples_summary["ripple_CAR_check"] = CAR_check
    ripples_summary["ripple_SRP_check"] = SRP_check
    ripples_summary["ripple_CAR_check_lr"] = CAR_check_lr
    ripples_summary["ripple_SRP_check_lr"] = SRP_check_lr

    session: Session = Session(
        ripples_summary=RipplesSummary(**ripples_summary),
        clusters_info=clusters_info,
        id=metadata_probe["Session"],
        length_seconds=lfp.shape[1] / sampling_rate_lfp,
        rms_per_channel=rms_per_channel,
        sampling_rate_lfp=sampling_rate_lfp,
        CA1_channels_analysed=detection_channels_ca1,
        CA1_channels_swr_pow=CA1_channels_swr_pow,
    )

    with open(
        HERE.parent / "results" / "test_1902_5Median" / f"{recording_id}.json",
        "w",
    ) as f:
        json.dump(session.model_dump(), f)


def main() -> None:

    reprocess = True
    metadata = gsheet2df("1HSERPbm-kDhe6X8bgflxvTuK24AfdrZJzbdBy11Hpcg", "sessions", 1)
    metadata = metadata[metadata["test_cohort"] == "TRUE"]
    metadata = metadata[metadata["Ignore"] == "FALSE"]
    metadata = metadata[metadata["Perfect_Peak"] == "Definitely"]

    sessions_keep = []
    for session in metadata["Session"].to_list():
        if session[-2:] in ["3M", "4M"]:
            animal_name = "_".join(session.split("_")[:3])
            assert animal_name.startswith("WT") or animal_name.startswith("NLGF")
            sessions_keep.append(session)

    for _, row in metadata.iterrows():
        json_path = (
            HERE.parent
            / "results"
            / "test_1902_5Median"
            / f"{row['Session']}-{row['Recording Name']}-Probe{row['Probe']}.json"
        )

        if json_path.exists() and not reprocess:
            print(f"Skipping {json_path}")
            continue
        print(f"Processing {json_path}")
        cache_session(row)
