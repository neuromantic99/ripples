from pathlib import Path
from typing import List, Tuple, Any
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from scipy import stats, ndimage
import pandas as pd
import statsmodels.formula.api as smf

from ripples.consts import RIPPLE_BAND, RESULTS_PATH
from ripples.models import (
    ClusterType,
    Session,
    SessionToAverage,
    CandidateEvent,
)
from ripples.waveforms import do_clustering_and_plot

from ripples.utils import (
    mean_across_same_session,
    bandpass_filter,
    compute_envelope,
    mean_across_sessions,
)
import pywt
from scipy import signal


def do_clustering_and_save_cell_type(
    WTs: List[Session], NLGFs: List[Session], area_map: dict
) -> tuple[List[Session], List[Session], pd.DataFrame]:

    all_data = pd.DataFrame()
    all_genotypes = WTs + NLGFs
    WTs = []
    NLGFs = []

    num_clusters_per_session = [len(session.clusters_info) for session in all_genotypes]
    all_data["Cluster Index"] = np.arange(sum(num_clusters_per_session))
    all_data["Genotype"] = np.hstack(
        [
            [all_genotypes[idx].id.split("_")[:3][0]] * num_clusters_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    ).astype(str)
    all_data["ID"] = np.hstack(
        [
            np.full(
                num_clusters_per_session[idx],
                int(all_genotypes[idx].id.split("_")[:3][2]),
            )
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["unique_ID"] = np.hstack(
        [
            [
                (
                    all_genotypes[idx].id.split("_")[:3][2]
                    + "_"
                    + all_genotypes[idx].baseline
                    + "_"
                    + all_genotypes[idx].id.split("_")[:4][3]
                )
            ]
            * num_clusters_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["timepoint"] = np.hstack(
        [
            [all_genotypes[idx].id.split("_")[:4][3]] * num_clusters_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["region"] = np.hstack(
        [
            cluster.region[0:3]
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )

    all_data["channel"] = np.hstack(
        [
            cluster.channel
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["depth"] = np.hstack(
        [
            cluster.depth
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["ks_label"] = np.hstack(
        [
            cluster.info
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["quality_metrics"] = np.hstack(
        [
            cluster.good_cluster
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["spike_times"] = [
        cluster.spike_times
        for idx in range(len(all_genotypes))
        for cluster in all_genotypes[idx].clusters_info
    ]
    all_data["aligned_waveforms"] = [
        np.array(cluster.aligned_wf[0])
        for idx in range(len(all_genotypes))
        for cluster in all_genotypes[idx].clusters_info
    ]

    all_data["fullwidth_at_third_max"] = np.hstack(
        [
            [cluster.halfwidth_at_third_max]
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["valley_to_peak_time"] = np.hstack(
        [
            [cluster.valley_to_peak_time]
            for idx in range(len(all_genotypes))
            for cluster in all_genotypes[idx].clusters_info
        ]
    )
    all_data["cell_type"] = np.full(len(all_data), "", dtype=str)
    for area in area_map:
        all_data_new = all_data[all_data["region"] == area.upper()]
        cell_type = do_clustering_and_plot(all_data_new)
        all_data.loc[all_data["region"] == area.upper(), "cell_type"] = cell_type
    plt.show()

    for session in all_genotypes:
        session_df = all_data[all_data["unique_ID"] == session.unique_id]
        for cluster_idx, cluster in enumerate(session.clusters_info):
            assert cluster.spike_times == session_df.loc[cluster_idx, "spike_times"]
            cluster.cell_type = session_df.loc[cluster_idx, "cell_type"]

    return WTs, NLGFs, all_data


def count_spikes_around_ripple_simple(
    peak_time: float,
    spike_times: np.ndarray,
    padding: float,
    num_bins: int,
) -> np.ndarray:

    spike_times_around_ripple = spike_times[
        np.logical_and(
            spike_times > (peak_time - padding),
            spike_times < (peak_time + padding),
        )
    ]

    counts, _ = np.histogram(
        spike_times_around_ripple,
        bins=num_bins,
        range=(peak_time - padding, peak_time + padding),
    )

    return np.array(counts)


def count_spikes_around_ripple_grouped(
    peak_time: float,
    onset: float,
    offset: float,
    spike_times: List[np.ndarray],  # List of clusters with their spiketimes
    padding: float,
    num_bins: int,
    group: str,
) -> np.ndarray:

    spiking_response_per_ripple_grouped = []
    modulation = np.zeros(len(spike_times))
    if group == "all_cells":
        spiking_response_per_ripple_grouped = spike_times
    else:
        around_ripple_time = 0.7
        ripple_duration = offset - onset
        for cluster in range(len(spike_times)):
            spike_times_cluster = spike_times[cluster]

            during_ripple = np.logical_and(
                spike_times_cluster > onset, spike_times_cluster < offset
            )
            around_ripple = np.logical_and(
                spike_times_cluster > onset - 1, spike_times_cluster < onset - 0.3
            )

            firing_during_ripple = during_ripple.sum() / ripple_duration
            firing_around_ripple = around_ripple.sum() / around_ripple_time

            if firing_around_ripple > 0:
                modulation[cluster] = (
                    firing_during_ripple - firing_around_ripple
                ) / firing_around_ripple
                if group == "pos_responders":
                    if modulation[cluster] > 0.5:
                        spiking_response_per_ripple_grouped.append(spike_times_cluster)
                if group == "neg_responders":
                    if modulation[cluster] < -0.5:
                        spiking_response_per_ripple_grouped.append(spike_times_cluster)
            elif np.logical_and(firing_around_ripple == 0, firing_during_ripple > 0):
                firing_around_ripple = 1
                modulation[cluster] = (
                    firing_during_ripple - firing_around_ripple
                ) / firing_around_ripple
                if group == "pos_responders":
                    if modulation[cluster] > 0.5:
                        spiking_response_per_ripple_grouped.append(spike_times_cluster)
                if group == "neg_responders":
                    if modulation[cluster] < -0.5:
                        spiking_response_per_ripple_grouped.append(spike_times_cluster)

    if not spiking_response_per_ripple_grouped:
        mean_spiking_response_per_ripple_grouped = np.empty(num_bins)
        mean_spiking_response_per_ripple_grouped[:] = None
    else:
        all_data = np.hstack(spiking_response_per_ripple_grouped)
        mean_spiking_response_per_ripple_grouped = count_spikes_around_ripple_simple(
            peak_time, all_data, padding, num_bins
        )
    return mean_spiking_response_per_ripple_grouped


def calculate_spiking_response(session: Session, ind_area: List[int]) -> np.ndarray:

    num_ripples = len(
        [
            event
            for event in session.ripples_summary.events
            if event
            if len(event.raw_lfp) == 5000
        ]
    )
    all_ripple_times = np.zeros(num_ripples)
    all_ripple_onset_times = np.zeros(num_ripples)
    all_ripple_offset_times = np.zeros(num_ripples)

    n = -1
    for event in range(len(session.ripples_summary.events)):
        if session.ripples_summary.events[event]:
            if len(session.ripples_summary.events[event].raw_lfp) == 5000:
                n = n + 1
                all_ripple_times[n] = (
                    session.ripples_summary.events[event].peak_idx
                    / session.sampling_rate_lfp
                )
                all_ripple_onset_times[n] = (
                    session.ripples_summary.events[event].onset
                    / session.sampling_rate_lfp
                )
                all_ripple_offset_times[n] = (
                    session.ripples_summary.events[event].offset
                    / session.sampling_rate_lfp
                )
    assert num_ripples == len(all_ripple_onset_times)

    response_per_ripple = np.zeros(num_ripples)
    ripple_duration = all_ripple_offset_times - all_ripple_onset_times
    area_clusters = [
        session.clusters_info[cluster]
        for cluster in range(len(session.clusters_info))
        if cluster in ind_area
    ]
    num_clusters = len(area_clusters)
    response = np.zeros((num_ripples, num_clusters), dtype=bool)
    response_percent_change = np.zeros((num_ripples, num_clusters))

    for cluster_idx in range(num_clusters):
        spike_times = np.array(area_clusters[cluster_idx].spike_times)
        for ripple_ind in range(num_ripples):

            during_ripple = np.logical_and(
                spike_times > all_ripple_onset_times[ripple_ind],
                spike_times < all_ripple_offset_times[ripple_ind],
            )
            around_ripple = np.logical_and(
                spike_times > (all_ripple_times[ripple_ind] - 1),
                spike_times < (all_ripple_times[ripple_ind] - 0.3),
            )
            around_ripple_time = 0.7
            firing_during_ripple = during_ripple.sum() / ripple_duration[ripple_ind]
            firing_around_ripple = around_ripple.sum() / around_ripple_time

            if firing_around_ripple > 0:
                response_percent_change[ripple_ind, cluster_idx] = (
                    firing_during_ripple - firing_around_ripple
                ) / firing_around_ripple
            elif np.logical_and(firing_around_ripple == 0, firing_during_ripple > 0):
                firing_around_ripple = 1
                response_percent_change[ripple_ind, cluster_idx] = (
                    firing_during_ripple - firing_around_ripple
                ) / firing_around_ripple
            else:
                response_percent_change[ripple_ind, cluster_idx] = 0
            response[ripple_ind, cluster_idx] = np.logical_or(
                response_percent_change[ripple_ind, cluster_idx] > 0.5,
                response_percent_change[ripple_ind, cluster_idx] < -0.5,
            )

    # calculate for each ripple percentage of cells responding with a 20% change in firing rate during the ripple compared to the firing around the ripple
    response_per_ripple = sum(response.T) / num_clusters
    percentage_increasing_cells = np.array(
        [sum(ripple > 0.5) / num_clusters for ripple in response_percent_change]
    )
    percentage_decreasing_cells = np.array(
        [sum(ripple < -0.5) / num_clusters for ripple in response_percent_change]
    )
    percentage_non_responding_cells = np.array(
        [
            sum(np.logical_and(ripple <= 0.5, ripple >= -0.5)) / num_clusters
            for ripple in response_percent_change
        ]
    )
    percentages_response = [
        [
            percentage_increasing_cells[ripple_idx],
            percentage_decreasing_cells[ripple_idx],
            percentage_non_responding_cells[ripple_idx],
        ]
        for ripple_idx in range(num_ripples)
    ]

    result = np.hstack(
        (response_per_ripple.reshape(num_ripples, 1), np.array(percentages_response))
    )

    return result


def get_clusters_ind_per_region(session: Session, area: str) -> List:
    cluster_region = [cluster.region for cluster in session.clusters_info]
    clusters_keep = [
        idx
        for idx, region in enumerate(cluster_region)
        if region is not None and area in region.lower()
    ]
    return clusters_keep


def get_good_clusters_only(
    session: Session, thresholding: str = "quality_metrics"
) -> Session:

    if thresholding == "quality_metrics":
        session.clusters_info = [
            session.clusters_info[cluster]
            for cluster in range(len(session.clusters_info))
            if session.clusters_info[cluster].good_cluster
        ]

    elif thresholding == "kilosort":
        session.clusters_info = [
            session.clusters_info[cluster]
            for cluster in range(len(session.clusters_info))
            if session.clusters_info[cluster].info == ClusterType.GOOD
        ]

    return session


def build_dataframe_per_ripple(
    WTs: List[Session],
    NLGFs: List[Session],
    area_map: dict,
    groups: List[str],
    padding_in_seconds: float,
) -> pd.DataFrame:

    all_data = pd.DataFrame()
    all_genotypes = WTs + NLGFs
    WTs = []
    NLGFs = []

    num_ripples_per_session = [
        len(
            [
                event
                for event in session.ripples_summary.events
                if event
                if len(event.raw_lfp) == 5000
            ]
        )
        for session in all_genotypes
    ]
    all_data["Ripple Index"] = np.arange(sum(num_ripples_per_session))
    all_data["Genotype"] = np.hstack(
        [
            [all_genotypes[idx].id.split("_")[:3][0]] * num_ripples_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    ).astype(str)
    all_data["ID"] = np.hstack(
        [
            np.full(
                num_ripples_per_session[idx],
                int(all_genotypes[idx].id.split("_")[:3][2]),
            )
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["unique_ID"] = np.hstack(
        [
            [
                (
                    all_genotypes[idx].id.split("_")[:3][2]
                    + "_"
                    + all_genotypes[idx].baseline
                    + "_"
                    + all_genotypes[idx].id.split("_")[:4][3]
                )
            ]
            * num_ripples_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["timepoint"] = np.hstack(
        [
            [all_genotypes[idx].id.split("_")[:4][3]] * num_ripples_per_session[idx]
            for idx in range(len(all_genotypes))
        ]
    )
    all_data["Amplitude"] = np.hstack(
        [
            np.array(
                [
                    event.peak_amplitude
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Strength"] = np.hstack(
        [
            np.array(
                [
                    event.strength
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Frequency"] = np.hstack(
        [
            np.array(
                [
                    event.frequency
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Duration"] = np.hstack(
        [
            np.array(
                [
                    (event.offset - event.onset) / session.sampling_rate_lfp * 1000
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Raw_LFP"] = np.vstack(
        [
            np.array(
                [
                    event.raw_lfp
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    ).tolist()
    all_data["Peak_idx"] = np.hstack(
        [
            np.array(
                [
                    event.peak_idx
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Onset"] = np.hstack(
        [
            np.array(
                [
                    event.onset
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_data["Offset"] = np.hstack(
        [
            np.array(
                [
                    event.offset
                    for event in session.ripples_summary.events
                    if event
                    if len(event.raw_lfp) == 5000
                ]
            )
            for session in all_genotypes
        ]
    )
    all_genotypes = [
        get_good_clusters_only(session, thresholding="quality_metrics")
        for session in all_genotypes
    ]

    for area in area_map:
        ind_area = [
            get_clusters_ind_per_region(session, area) for session in all_genotypes
        ]
        spike_times_per_session = [
            [
                np.array(cluster.spike_times)
                for cluster in all_genotypes[session_idx].clusters_info[
                    min(ind_area[session_idx]) : max(ind_area[session_idx])
                ]
            ]
            for session_idx in range(len(all_genotypes))
        ]

        response_per_ripple = np.vstack(
            [
                calculate_spiking_response(
                    all_genotypes[session_idx], ind_area[session_idx]
                )
                for session_idx in range(len(all_genotypes))
            ]
        )

        all_data["Percentage_all_resp_" + area_map[area]] = response_per_ripple[:, 0]
        all_data["Percentage_pos_resp_" + area_map[area]] = response_per_ripple[:, 1]
        all_data["Percentage_neg_resp_" + area_map[area]] = response_per_ripple[:, 2]
        all_data["Percentage_non_resp_" + area_map[area]] = response_per_ripple[:, 3]

        num_bins = int(padding_in_seconds * 2 * 100)  # 10ms bins

        for group in groups:
            all_data["Spiking_response_" + area_map[area] + "_" + group] = np.vstack(
                [
                    [
                        count_spikes_around_ripple_grouped(
                            event.peak_idx
                            / all_genotypes[session_idx].sampling_rate_lfp,
                            event.onset / all_genotypes[session_idx].sampling_rate_lfp,
                            event.offset / all_genotypes[session_idx].sampling_rate_lfp,
                            spike_times_per_session[session_idx],
                            padding=padding_in_seconds,
                            num_bins=num_bins,
                            group=group,
                        )
                        for event in all_genotypes[session_idx].ripples_summary.events
                        if event
                        if len(event.raw_lfp) == 5000
                    ]
                    for session_idx in range(len(all_genotypes))
                ]
            ).tolist()

            all_data["Peak_response_" + area_map[area] + "_" + group] = [
                element[int(padding_in_seconds * num_bins)]
                for element in np.array(
                    all_data["Spiking_response_" + area_map[area] + "_" + group]
                )
            ]
    return all_data


def plot_heatmap_firing_around_ripple_per_session(
    all_data: pd.DataFrame, area: str, padding_seconds: float
) -> None:
    session_id = all_data["unique_ID"].unique()
    for session in session_id:
        x_axis = np.linspace(
            -padding_seconds, padding_seconds, int(padding_seconds * 2 * 100)
        )
        ripple_num = len(all_data[all_data["unique_ID"] == session])
        y_axis = np.linspace(1, ripple_num + 1, ripple_num)
        X, Y = np.meshgrid(x_axis, y_axis)
        plt.figure()
        plt.pcolor(
            X,
            Y,
            [
                np.array(all_data["Spiking_response_" + area + "_all_cells"][ripple])
                for ripple in range(len(all_data))
                if all_data["unique_ID"][ripple] == session
                if not np.isnan(
                    sum(all_data["Spiking_response_" + area + "_all_cells"][ripple])
                )
            ],
        )
        plt.title(area + session)
        plt.xlabel("time (sec)")
        plt.ylabel("Ripple Number")
    plt.show()


def plot_ripple_triggered_spikes(
    data: np.ndarray[any, any],
    region: str,
    color: str,
    session_id: List[str],
    padding_seconds: float,
) -> None:

    data = mean_across_sessions(
        [
            SessionToAverage(session_id[session], data[session])
            for session in range(len(session_id))
        ]
    )

    x_axis = np.linspace(
        -padding_seconds, padding_seconds, int(padding_seconds * 2 * 100)
    )

    plt.plot(
        x_axis,
        np.mean(data, axis=0),
        color=color,
        label=region,
    )

    plt.fill_between(
        x_axis,
        np.mean(data, axis=0) - np.std(data, axis=0) / np.sqrt(data.shape[0]),
        np.mean(data, axis=0) + np.std(data, axis=0) / np.sqrt(data.shape[0]),
        color=color,
        alpha=0.3,
    )


def smooth_ripple_triggered_average_gaussian(stacked_trials: np.ndarray) -> np.ndarray:
    stacked_trials = stacked_trials[~np.isnan(np.mean(stacked_trials, axis=1))]
    filtered_signal = ndimage.gaussian_filter1d(stacked_trials, 0.42, axis=0)
    return filtered_signal


def plot_firing_around_ripple_MUA_per_ripple_first(
    all_data: pd.DataFrame, area_map: dict, groups: List[str], padding_seconds: float
) -> None:
    anims = all_data["unique_ID"].unique()
    genotypes = np.full(len(anims), str)
    anim_id = np.full(len(anims), str)

    for group in groups:
        plt.figure()
        plt.suptitle(group)
        n = 0
        for area in area_map:
            n = n + 1
            ind = -1
            mean_ripple_response = np.full((len(anims), 100), 0, dtype=float)
            for anim in anims:
                ind = ind + 1
                genotypes[ind] = np.array(
                    all_data["Genotype"][all_data["unique_ID"] == anim]
                )[0]
                anim_id[ind] = np.array(all_data["ID"][all_data["unique_ID"] == anim])[
                    0
                ]

                mean_ripple_response[ind, :] = np.nanmean(
                    stats.zscore(
                        smooth_ripple_triggered_average_gaussian(
                            np.vstack(
                                np.array(
                                    all_data[
                                        "Spiking_response_"
                                        + area_map[area]
                                        + "_"
                                        + group
                                    ]
                                )[np.array(all_data["unique_ID"] == anim)]
                            ),
                        ),
                        axis=1,
                    ),
                    axis=0,
                )

            assert len(mean_ripple_response) == len(genotypes)
            plt.subplot(1, 3, n)
            plt.title(area_map[area])
            plot_ripple_triggered_spikes(
                mean_ripple_response[genotypes == "WT"],
                area_map[area],
                "blue",
                anim_id[genotypes == "WT"],
                padding_seconds=padding_seconds,
            )
            plot_ripple_triggered_spikes(
                mean_ripple_response[genotypes == "NLGF"],
                area_map[area],
                "red",
                anim_id[genotypes == "NLGF"],
                padding_seconds=padding_seconds,
            )
            plt.ylabel("Firing Rate (z-score)")
            plt.xlabel("Time from ripple (s)")


def plot_distributions_violins_anim_means_per_characteristic(
    all_data: pd.DataFrame,
) -> None:
    characteristics = [
        "Amplitude",
        "Strength",
        "Frequency",
        "Duration",
        "Percentage_all_resp_ca1",
        "Percentage_all_resp_retrosplenial",
        "Percentage_all_resp_dentate",
        "Percentage_pos_resp_ca1",
        "Percentage_pos_resp_retrosplenial",
        "Percentage_pos_resp_dentate",
        "Percentage_neg_resp_ca1",
        "Percentage_neg_resp_retrosplenial",
        "Percentage_neg_resp_dentate",
        "Percentage_non_resp_ca1",
        "Percentage_non_resp_retrosplenial",
        "Percentage_non_resp_dentate",
        "Peak_response_ca1_neg_responders",
        "Peak_response_retrosplenial_neg_responders",
        "Peak_response_dentate_neg_responders",
        "Peak_response_ca1_pos_responders",
        "Peak_response_retrosplenial_pos_responders",
        "Peak_response_dentate_pos_responders",
        "Peak_response_ca1_all_cells",
        "Peak_response_retrosplenial_all_cells",
        "Peak_response_dentate_all_cells",
    ]
    ylabels = [
        "$\mu$V",
        "AU",
        "Hz",
        "ms",
        "AU",
        "AU",
        "AU",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Percent",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
        "Mean firing rate [Hz]",
    ]

    for plots in range(len(characteristics)):
        plt.figure()
        plt.suptitle(characteristics[plots])
        sns.kdeplot(data=all_data, x=characteristics[plots], hue="Genotype")

        model = smf.mixedlm(
            characteristics[plots] + " ~ Genotype",
            data=all_data,
            groups=all_data["ID"],
            use_sqrt=True,
            missing="drop",
        )
        mdf = model.fit()

        plt.figure()
        plt.suptitle(characteristics[plots])
        plt.subplot(1, 2, 1)
        plt.title(
            "LME-Model p-value:" + str(mdf.pvalues["Genotype[T.WT]"]),
            fontdict={"fontsize": 7},
        )
        ax = sns.violinplot(
            data=all_data,
            x="Genotype",
            y=str(characteristics[plots]),
            fill=False,
            inner=None,
        )
        colors = ["blue", "red"]
        for ind, violin in enumerate(ax.findobj(PolyCollection)):
            rgb_edge = to_rgb(colors[ind])
            rgb = 0.5 + 0.5 * np.array(rgb_edge)
            violin.set_facecolor(rgb)
            violin.set_edgecolor(rgb_edge)
        # sns.swarmplot(
        #     data=all_data,
        #     x="Genotype",
        #     y=str(characteristics[plots]),
        #     size=0.5,
        #     color="black",
        # )
        plt.ylabel(ylabels[plots])
        plt.legend()
        plt.tight_layout()

        WT_data = (
            all_data[all_data["Genotype"] == "WT"]
            .groupby(["ID", "unique_ID"])[characteristics[plots]]
            .mean()
            .groupby("ID")
            .mean()
            .dropna()
        ).to_numpy()

        NLGF_data = (
            all_data[all_data["Genotype"] == "NLGF"]
            .groupby(["ID", "unique_ID"])[characteristics[plots]]
            .mean()
            .groupby("ID")
            .mean()
            .dropna()
        ).to_numpy()

        all_data_to_plot = [WT_data, NLGF_data]

        shapiro_result_WT = stats.shapiro(WT_data)
        shapiro_result_NLGF = stats.shapiro(NLGF_data)
        if shapiro_result_WT.pvalue < 0.05 * shapiro_result_NLGF.pvalue < 0.05:
            stat_results = stats.ttest_ind(
                WT_data,
                NLGF_data,
            )
            test = "t-test"
        else:
            test = "Mann-Whitney-U test"
            stat_results = stats.mannwhitneyu(
                WT_data,
                NLGF_data,
            )
        plt.subplot(1, 2, 2)
        plt.title(
            +", " + test + ", p-value=" + str(stat_results.pvalue),
            fontdict={"fontsize": 7},
        )
        plt.boxplot(
            all_data_to_plot,
            tick_labels=[
                "WT(n=" + str(len(WT_data)) + ")",
                "NLGF(n=" + str(len(NLGF_data)) + ")",
            ],
        )
        plt.tight_layout


def plot_correlations(
    all_data_new: pd.DataFrame,
    characteristics_ripples: List[str],
    characteristics_response: List[str],
) -> None:

    for ind in range(len(characteristics_ripples)):
        plt.figure()
        for ind_resp in range(len(characteristics_response)):
            corr_WTs = stats.pearsonr(
                x=all_data_new[characteristics_ripples[ind]][
                    all_data_new["Genotype"] == "WT"
                ],
                y=all_data_new[characteristics_response[ind_resp]][
                    all_data_new["Genotype"] == "WT"
                ],
            )
            corr_NLGFs = stats.pearsonr(
                x=all_data_new[characteristics_ripples[ind]][
                    all_data_new["Genotype"] == "NLGF"
                ],
                y=all_data_new[characteristics_response[ind_resp]][
                    all_data_new["Genotype"] == "NLGF"
                ],
            )
            # plt.subplot(2, 3, ind_resp + 1)
            sns.lmplot(
                x=characteristics_ripples[ind],
                y=characteristics_response[ind_resp],
                hue="Genotype",
                data=all_data_new,
                markers=".",
                legend=False,
            )

            plt.title(
                "p_WT:"
                + str(round(corr_WTs.pvalue, ndigits=6))
                + "corr:"
                + str(round(corr_WTs.correlation, ndigits=4))
                + "p_NLGF:"
                + str(round(corr_NLGFs.pvalue, ndigits=6))
                + "corr:"
                + str(round(corr_NLGFs.correlation, ndigits=4)),
                fontdict={"fontsize": 12},
            )
            plt.tight_layout()
            plt.legend(fontsize=8)

    plt.show()


def count_spikes_around_ripple(
    ripple_peak_times: np.ndarray,
    spike_times: List[float],
    padding: float,
    num_bins: int,
) -> np.ndarray:

    counts_per_cluster = np.empty((len(ripple_peak_times), num_bins))
    spike_times_cluster = np.array(spike_times)
    for idx in range(len(ripple_peak_times)):
        peak_time = ripple_peak_times[idx]
        spike_times_around_ripple = spike_times_cluster[
            np.logical_and(
                spike_times_cluster > (peak_time - padding),
                spike_times_cluster < (peak_time + padding),
            )
        ]

        counts, _ = np.histogram(
            spike_times_around_ripple,
            bins=num_bins,
            range=(peak_time - padding, peak_time + padding),
        )
        counts_per_cluster[idx] = counts.astype(float)

    mean_firing_around_ripples = np.mean(counts_per_cluster, axis=0)

    return mean_firing_around_ripples


def mean_across_ripples(
    session: Session,
    padding: float,
    area_map: dict,
    plot: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    all_ripple_times = np.array(
        [
            event.peak_idx / session.sampling_rate_lfp
            for event in session.ripples_summary.events
            if event
        ]
    )
    mean_firing_around_ripples = np.empty((len(session.clusters_info), 100))
    for cluster in range(len(session.clusters_info)):
        mean_firing_around_ripples[cluster] = count_spikes_around_ripple(
            all_ripple_times,
            session.clusters_info[cluster].spike_times,
            padding=padding,
            num_bins=int(padding * 2 * 100),
        )

    if smooth:
        mean_firing_around_ripples = smooth_ripple_triggered_average_gaussian(
            mean_firing_around_ripples
        )

    if plot:
        for area in area_map:
            ind_area = get_clusters_ind_per_region(session, area)
            plt.figure()
            plt.pcolormesh(
                stats.zscore(
                    mean_firing_around_ripples[min(ind_area) : max(ind_area)], axis=1
                )
            )
            plt.title(session.id + ", " + session.baseline + ", " + area_map[area])
            plt.show()

    return np.array(mean_firing_around_ripples)


def plot_ripple_triggered_firing_per_cluster_first(
    WTs: List[Session],
    NLGFs: List[Session],
    area_map: dict,
    padding: float,
    good_units_only: bool = True,
) -> None:

    if good_units_only:
        WTs = [
            get_good_clusters_only(session, thresholding="quality_metrics")
            for session in WTs
        ]
        NLGFs = [
            get_good_clusters_only(session, thresholding="quality_metrics")
            for session in NLGFs
        ]
        assert len(WTs[0].clusters_info) == sum(
            [cluster.good_cluster for cluster in WTs[0].clusters_info]
        )

    mean_firing_around_ripples_all_clusters_WTs = [
        mean_across_ripples(session, padding, area_map, plot=False) for session in WTs
    ]
    mean_firing_around_ripples_all_clusters_NLGFs = [
        mean_across_ripples(session, padding, area_map, plot=False) for session in NLGFs
    ]

    fig = plt.figure(figsize=(4 * 3, 4))
    n = 0
    for area in area_map:
        n = n + 1
        # using the nan mean because z_scoring for silent cells without firing around ripples will give back nans
        mean_per_session_WTs = np.vstack(
            [
                np.nanmean(
                    stats.zscore(
                        mean_firing_around_ripples_all_clusters_WTs[session][
                            get_clusters_ind_per_region(WTs[session], area)
                        ],
                        axis=1,
                    ),
                    axis=0,
                )
                for session in range(len(WTs))
            ]
        )

        session_id_WTs = [session.id for session in WTs]

        mean_per_session_NLGFs = np.vstack(
            [
                np.nanmean(
                    stats.zscore(
                        mean_firing_around_ripples_all_clusters_NLGFs[session][
                            get_clusters_ind_per_region(NLGFs[session], area)
                        ],
                        axis=1,
                    ),
                    axis=0,
                )
                for session in range(len(NLGFs))
            ]
        )

        session_id_NLGFs = [session.id for session in NLGFs]

        plt.subplot(1, 3, n)
        plot_ripple_triggered_spikes(
            mean_per_session_WTs,
            area,
            "blue",
            session_id_WTs,
            padding_seconds=padding,
        )
        plot_ripple_triggered_spikes(
            mean_per_session_NLGFs,
            area,
            "red",
            session_id_NLGFs,
            padding_seconds=padding,
        )

        plt.ylabel("Firing Rate (z-score)")
        plt.xlabel("Time from ripple (s)")
        plt.title(area_map[area])
        fig.legend(["WT", "_", "NLGF"])
        fig.suptitle("Averaging across ripples per cluster first")


def plot_ripple_raw(
    ripple: CandidateEvent,
    sampling_rate: float | None,
    session_id: str | None,
    rec_ind: str | None,
    ripple_ind: str | None,
) -> Any:
    raw = np.array(ripple.raw_lfp)
    raw = raw[2000:-2000]
    if sampling_rate:
        sampling_rate = sampling_rate
    else:
        sampling_rate = 2500
    filtered = bandpass_filter(
        raw.reshape(1, len(raw)),
        RIPPLE_BAND[0],
        RIPPLE_BAND[1],
        sampling_rate,
    )
    envelope = signal.savgol_filter(compute_envelope(filtered), 101, 4)
    filtered = filtered.reshape(filtered.shape[1], 1)
    envelope = envelope.reshape(envelope.shape[1], 1)

    all_data = np.hstack((filtered, envelope))

    fig, axs = plt.subplots(3, 1, sharex=True)
    time = range(-500, 500, 1)

    fs = sampling_rate
    w = 4.0
    freq = np.linspace(1, 500, 100)
    scales = w * fs / (2 * freq * np.pi)
    scales = np.logspace(np.log10(scales.min()), np.log10(scales.max()), num=100)
    cwtm, freqs = pywt.cwt(raw, scales, "cmor2.5-1.5", sampling_period=1 / fs)
    cwtm /= np.max(np.abs(cwtm))
    axs[0].pcolormesh(
        time,
        freqs[0:50],
        np.abs(cwtm[0:50, :]),
        cmap="viridis",
        shading="gouraud",
    )
    axs[0].axhline(RIPPLE_BAND[0], color="red", linestyle="-")
    axs[0].axhline(RIPPLE_BAND[1], color="red", linestyle="-")
    font = {"size": 10}

    onset_time = ripple.onset
    offset_time = ripple.offset
    peak_time = ripple.peak_idx

    # Plot
    axs[1].plot(time, raw, linewidth=0.5)
    axs[2].plot(time, all_data, linewidth=0.5)
    axs[2].axvline((onset_time - peak_time), color="red", linestyle="-")
    axs[2].axvline((offset_time - peak_time), color="red", linestyle="-")
    axs[2].set_xlabel("time (ms)")
    axs[2].set_title(
        f"onset: {onset_time/sampling_rate}",
        fontdict=font,
    )
    plt.show()
    plt.tight_layout()

    if session_id:
        figure_path = RESULTS_PATH / "figures" / "ripples" / session_id / rec_ind
        if not figure_path.exists():
            os.makedirs(figure_path)

        plt.savefig(figure_path / ripple_ind)


def plot_ripple_raw_session(all_mice: List[Session]) -> None:

    for rec in range(len(all_mice)):
        idx = range(len(all_mice[rec].ripples_summary.events))
        for n in idx:
            if not all_mice[rec].ripples_summary.events[n]:
                continue
            if len(all_mice[rec].ripples_summary.events[n].raw_lfp) < 5000:
                continue
            else:
                ripple = all_mice[rec].ripples_summary.events[n]
                session_id = f"{all_mice[rec].id}"
                rec_ind = f"{all_mice[rec].baseline}"
                ripple_ind = f"ripple_{n}.png"
                plot_ripple_raw(
                    ripple,
                    all_mice[rec].sampling_rate_lfp,
                    session_id,
                    rec_ind,
                    ripple_ind,
                )


def plot_ripple_frequency_spectrum(all_mice: List[Session]) -> None:
    for rec in range(len(all_mice)):
        idx = range(len(all_mice[rec].ripples_summary.events))
        for n in idx:
            if not all_mice[rec].ripples_summary.events[n]:
                continue
            if not all_mice[rec].ripples_summary.events[n].raw_lfp:
                continue
            else:
                onset_time = all_mice[rec].ripples_summary.events[n].onset
                offset_time = all_mice[rec].ripples_summary.events[n].offset
                peak_time = all_mice[rec].ripples_summary.events[n].peak_idx

                raw = np.array(all_mice[rec].ripples_summary.events[n].raw_lfp)[
                    2000:-2000
                ]
                ripple = raw[
                    (500 + onset_time - peak_time) : (500 + offset_time - peak_time)
                ]
                [f, Pxx] = signal.periodogram(ripple, all_mice[rec].sampling_rate_lfp)
                max_idx = np.argmax(Pxx.reshape(len(f), 1).tolist())
                max_freq = f[max_idx]
                max_val = (Pxx.reshape(len(f), 1)).tolist()[max_idx]
                max_val = max_val[0]
                ev_freq = all_mice[rec].ripples_summary.events[n].frequency
                peaks, _ = signal.find_peaks(Pxx, height=0.25 * max_val)
                plt.figure()
                plt.plot(f, Pxx.reshape(len(f), 1))
                plt.plot(max_freq, max_val, "ro")
                plt.plot(f[peaks], Pxx[peaks], "bx")
                font = {"size": 10}
                plt.title(
                    f"Peak frequencies: {f[peaks]}; Ripple frequency: {ev_freq}",
                    fontdict=font,
                )
                figure_path = (
                    RESULTS_PATH
                    / "figures"
                    / "ripples_freq_spectrum"
                    / f"{all_mice[rec].id}"
                    / f"{rec}"
                )
                if not figure_path.exists():
                    os.makedirs(figure_path)

                plt.savefig(figure_path / f"ripple_{n}.png")
                plt.close()


def plot_single_ripples(WTs: List[Session], NLGFs: List[Session]) -> None:
    all_mice = WTs + NLGFs
    plot_ripple_raw_session(all_mice)
    plot_ripple_frequency_spectrum(all_mice)


def number_of_spikes_per_cell_per_ripple(session: Session) -> float:
    """Follows the Tonegawa approach: pubmed.ncbi.nlm.nih.gov/24139046/"""
    all_ripple_times = np.array(
        [
            event.peak_idx / session.sampling_rate_lfp
            for event in session.ripples_summary.events
            if event
        ]
    )

    spikes_per_ripple = np.zeros(len(session.clusters_info))

    for cluster_idx, cluster in enumerate(session.clusters_info):
        if cluster.info != ClusterType.GOOD or cluster.region != "CA1":
            continue

        spike_times = np.array(cluster.spike_times)
        ripple_starts = all_ripple_times - 0.3
        ripple_ends = all_ripple_times + 0.3

        # For each ripple time, check if spikes fall between start and end
        # Broadcasting: spike_times[:, None] creates a 2D array of spike_times against ripple_starts and ripple_ends
        within_ripple = (spike_times[:, None] > ripple_starts) & (
            spike_times[:, None] < ripple_ends
        )

        # Sum spikes within each ripple window
        spikes_per_ripple[cluster_idx] += np.sum(within_ripple)

    # Spikes per ripple is a vector of len n_clusters with the total number of times each cluster spiked during a ripple
    spikes_per_ripple = spikes_per_ripple / len(all_ripple_times)
    # Spikes per ripple is a vector of len n_clusters with the total number of times each cluster spiked per ripple
    spikes_per_ripple = spikes_per_ripple[spikes_per_ripple > 0]

    if len(spikes_per_ripple) == 0:
        return 0

    return np.mean(spikes_per_ripple).astype(float)


def spikes_per_ripple(WTs: List[Session], NLGFs: List[Session]) -> None:
    """(B) The number of spikes per SWR event, per cell
    (over all cells that fired at least one spike during at least one SWR event)."""

    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                session.id,
                number_of_spikes_per_cell_per_ripple(session),
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                session.id,
                number_of_spikes_per_cell_per_ripple(session),
            )
            for session in NLGFs
        ]
    )

    plt.figure()
    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data}, color="black")
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data}, showfliers=False)
    plt.ylabel("Spikes per ripple per CA1 neuron")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / "figures" / "spikes_per_ripple.png")
    plt.show()


def get_ripple_rate(session: Session) -> float:
    resting_seconds = (
        session.length_seconds * session.ripples_summary.resting_percentage
    )

    return (
        len([event.peak_idx for event in session.ripples_summary.events if event])
        / resting_seconds
    )


def resting_time_ripple_rate_correlation(
    WTs: List[Session], NLGFs: List[Session]
) -> None:
    plt.figure()
    all_mice = WTs + NLGFs
    plt.plot(
        [
            session.length_seconds * session.ripples_summary.resting_percentage
            for session in all_mice
        ],
        [get_ripple_rate(session) for session in all_mice],
        ".",
    )
    plt.savefig(RESULTS_PATH / "figures" / "resting_time_ripple_rate_correlation.png")


def plot_noise_levels(WTs: List[Session], NLGFs: List[Session]) -> None:
    plt.figure()

    wt_data = [
        np.nanmean(
            session.rms_per_channel[
                int(np.median(session.CA1_channels_analysed) - 10) : int(
                    np.median(session.CA1_channels_analysed) + 10
                )
            ]
        )
        for session in WTs
    ]
    ids = np.array([session.id for session in WTs])
    print("WT", ids)
    print("WT", wt_data)

    nlgf_data = [
        np.nanmean(
            session.rms_per_channel[
                int(np.median(session.CA1_channels_analysed) - 10) : int(
                    np.median(session.CA1_channels_analysed) + 10
                )
            ]
        )
        for session in NLGFs
    ]
    nIDs = np.array([session.id for session in NLGFs])
    print("NLGF", nIDs)
    print("NLGF", nlgf_data)

    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data}, color="black")
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data})
    plt.ylabel("Noise level (AU)")
    plt.tight_layout()
    plt.title("Mean RMS per channel for channels in CA1")
    plt.savefig(RESULTS_PATH / "figures" / "noise_level.png")


def filter_ripples(
    WTs: List[Session], NLGFs: List[Session]
) -> tuple[List[Session], List[Session]]:

    for session in range(len(WTs)):
        for n in range(len(WTs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(WTs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(WTs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(WTs[session].ripples_summary.ripple_freq_check[n])
                * np.array(WTs[session].ripples_summary.ripple_frequency[n] < 250)
            ) == False:
                WTs[session].ripples_summary.events[n] = []
    for session in range(len(NLGFs)):
        for n in range(len(NLGFs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(NLGFs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(NLGFs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(NLGFs[session].ripples_summary.ripple_freq_check[n])
                * np.array(NLGFs[session].ripples_summary.ripple_frequency[n] < 250)
            ) == False:
                NLGFs[session].ripples_summary.events[n] = []
    return WTs, NLGFs


def load_sessions(condition: List[str]) -> Tuple[List[Session], List[Session]]:

    results_files = Path(RESULTS_PATH).glob("*.json")
    WTs: List[Session] = []
    NLGFs: List[Session] = []

    for file in results_files:

        if condition[0] not in file.name or condition[1] not in file.name:
            continue

        with open(file, "r") as f:
            result = Session.model_validate_json(f.read())

        if not result.ripples_summary.ripple_amplitude:
            continue

        if "wt" in file.name.lower():
            WTs.append(result)
        elif "nlgf" in file.name.lower():
            NLGFs.append(result)
        else:
            raise ValueError(f"Unknown type of recording: {file.name}")

    return WTs, NLGFs


def main() -> None:

    condition = ["6M", "A"]
    WTs, NLGFs = load_sessions(condition)
    area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}
    cluster_dataframe, WTs, NLGFs = do_clustering_and_save_cell_type(
        WTs, NLGFs, area_map
    )
    plot_noise_levels(WTs, NLGFs)

    WTs, NLGFs = filter_ripples(WTs, NLGFs)
    plot_single_ripples(WTs, NLGFs)

    groups = ["pos_responders", "neg_responders", "all_cells"]
    area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}
    padding_firing_around_ripples = 0.5  # in seconds

    plot_ripple_triggered_firing_per_cluster_first(
        WTs, NLGFs, area_map, padding_firing_around_ripples, good_units_only=True
    )

    ripple_dataframe = build_dataframe_per_ripple(
        WTs, NLGFs, area_map, groups, padding_firing_around_ripples
    )
    plot_heatmap_firing_around_ripple_per_session(
        ripple_dataframe, area="ca1", padding_seconds=padding_firing_around_ripples
    )
    plot_firing_around_ripple_MUA_per_ripple_first(
        ripple_dataframe, area_map, groups, padding_firing_around_ripples
    )
    plot_distributions_violins_anim_means_per_characteristic(ripple_dataframe)

    characteristics_ripples = ["Strength", "Frequency", "Duration", "Amplitude"]
    characteristics_response = [
        "Peak_response_ca1_pos_responders",
        "Peak_response_retrosplenial_pos_responders",
        "Peak_response_dentate_pos_responders",
        "Peak_response_ca1_all_cells",
        "Peak_response_retrosplenial_all_cells",
        "Peak_response_dentate_all_cells",
        "Percentage_pos_resp_ca1",
        "Percentage_pos_resp_retrosplenial",
        "Percentage_pos_resp_dentate",
        "Percentage_neg_resp_ca1",
        "Percentage_neg_resp_retrosplenial",
        "Percentage_neg_resp_dentate",
        "Percentage_non_resp_ca1",
        "Percentage_non_resp_retrosplenial",
        "Percentage_non_resp_dentate",
    ]
    plot_correlations(
        ripple_dataframe, characteristics_ripples, characteristics_response
    )

    spikes_per_ripple(WTs, NLGFs)
    resting_time_ripple_rate_correlation(WTs, NLGFs)
