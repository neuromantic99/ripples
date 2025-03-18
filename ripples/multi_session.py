from pathlib import Path
from typing import List, Tuple, Any
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore
import pandas as pd

from ripples.consts import RIPPLE_BAND, RESULTS_PATH
from ripples.models import (
    ClusterType,
    Session,
    SessionToAverage,
)

from ripples.utils import (
    mean_across_same_session,
    bandpass_filter,
    compute_envelope,
    mean_across_sessions,
)
import pywt
from scipy import signal


sns.set_theme(context="talk", style="ticks")


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


def plot_ripple_raw(all_mice: List[Session]) -> None:

    for rec in range(len(all_mice)):
        idx = range(len(all_mice[rec].ripples_summary.events))
        for n in idx:
            if not all_mice[rec].ripples_summary.events[n]:
                continue
            if not all_mice[rec].ripples_summary.events[n].raw_lfp:
                continue
            raw = np.array(all_mice[rec].ripples_summary.events[n].raw_lfp)
            raw = raw[2000:-2000]
            filtered = bandpass_filter(
                raw.reshape(1, len(raw)),
                RIPPLE_BAND[0],
                RIPPLE_BAND[1],
                all_mice[rec].sampling_rate_lfp,
            )
            envelope = signal.savgol_filter(compute_envelope(filtered), 101, 4)
            filtered = filtered.reshape(filtered.shape[1], 1)
            envelope = envelope.reshape(envelope.shape[1], 1)

            all_data = np.hstack((filtered, envelope))

            fig, axs = plt.subplots(3, 1, sharex=True)
            time = range(-500, 500, 1)

            fs = all_mice[rec].sampling_rate_lfp
            w = 12.0
            freq = np.linspace(1, 500, 100)
            scales = w * fs / (2 * freq * np.pi)
            scales = np.logspace(
                np.log10(scales.min()), np.log10(scales.max()), num=100
            )
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
            axs[0].set_title(
                f"freq_check: {all_mice[rec].ripples_summary.ripple_freq_check[n]};CAR_check:{all_mice[rec].ripples_summary.ripple_CAR_check[n]};SRP_check:{all_mice[rec].ripples_summary.ripple_SRP_check[n]};CAR_check_lr:{all_mice[rec].ripples_summary.ripple_CAR_check_lr[n]};SRP_check_lr:{all_mice[rec].ripples_summary.ripple_SRP_check_lr[n]}",
                fontdict=font,
            )

            onset_time = all_mice[rec].ripples_summary.events[n].onset
            offset_time = all_mice[rec].ripples_summary.events[n].offset
            peak_time = all_mice[rec].ripples_summary.events[n].peak_idx

            # Plot
            axs[1].plot(time, raw, linewidth=0.5)
            axs[2].plot(time, all_data, linewidth=0.5)
            axs[2].axvline((onset_time - peak_time), color="red", linestyle="-")
            axs[2].axvline((offset_time - peak_time), color="red", linestyle="-")
            axs[2].set_xlabel("time (ms)")
            axs[2].set_title(
                f"onset: {onset_time/all_mice[rec].sampling_rate_lfp}",
                fontdict=font,
            )

            figure_path = (
                RESULTS_PATH / "figures" / "ripples" / f"{all_mice[rec].id}" / f"{rec}"
            )
            if not figure_path.exists():
                os.makedirs(figure_path)

            plt.savefig(figure_path / f"ripple_{n}.png")
            plt.close(fig)


def plot_ripple_frequency_spectrum(all_mice: List[Session]) -> None:
    for rec in range(len(all_mice)):
        idx = range(30)  # idx = range(len(all_mice[rec].ripples_summary.events))
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
    plot_ripple_raw(all_mice)
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


def plot_ripple_characteristics(WTs: List[Session], NLGFs: List[Session]) -> None:

    all_data = pd.DataFrame()
    all_genotypes = WTs + NLGFs

    _, anim_inds = np.unique(
        [session.id for session in all_genotypes], return_index=True
    )
    all_data["Genotype"] = np.array(
        [session.id.split("_")[0] for session in all_genotypes]
    )[anim_inds]
    all_data["ID"] = np.array([session.id.split("_")[2] for session in all_genotypes])[
        anim_inds
    ]
    all_data["timepoint"] = np.array(
        [session.id.split("_")[3] for session in all_genotypes]
    )[anim_inds]

    all_data["Mean Ripple Amplitude"] = mean_across_sessions(
        [
            SessionToAverage(
                session.id,
                np.mean(
                    [
                        event.peak_amplitude
                        for event in session.ripples_summary.events
                        if event
                    ]
                ).astype(float),
            )
            for session in all_genotypes
        ]
    )
    all_data["Mean Ripple Strength"] = mean_across_sessions(
        [
            SessionToAverage(
                session.id,
                np.mean(
                    [
                        event.strength
                        for event in session.ripples_summary.events
                        if event
                    ]
                ).astype(float),
            )
            for session in all_genotypes
        ]
    )
    all_data["Mean Ripple Frequency"] = mean_across_sessions(
        [
            SessionToAverage(
                session.id,
                np.mean(
                    [
                        event.frequency
                        for event in session.ripples_summary.events
                        if event
                    ]
                ).astype(float),
            )
            for session in all_genotypes
        ]
    )
    all_data["Mean Ripple Rate"] = mean_across_sessions(
        [
            SessionToAverage(session.id, get_ripple_rate(session))
            for session in all_genotypes
        ]
    )
    all_data["Mean Resting Time"] = mean_across_sessions(
        [
            SessionToAverage(
                session.id,
                session.length_seconds * session.ripples_summary.resting_percentage,
            )
            for session in all_genotypes
        ]
    )

    characteristics = [
        "Mean Ripple Amplitude",
        "Mean Ripple Strength",
        "Mean Ripple Frequency",
        "Mean Ripple Rate",
        "Mean Resting Time",
    ]
    ylabels = ["$\mu$V", "AU", "Hz", "Hz", "seconds"]

    for plots in range(len(characteristics)):
        plt.figure()
        plt.title(characteristics[plots])
        sns.stripplot(
            data=all_data, x="Genotype", y=str(characteristics[plots]), color="black"
        )
        sns.boxplot(
            data=all_data, x="Genotype", y=str(characteristics[plots]), showfliers=False
        )
        plt.ylabel(ylabels[plots])
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / "figures" / characteristics[plots])

    print("Done")

    # TODO: add in statistics


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


def smooth_ripple_triggered_average(
    stacked_trials: np.ndarray, bin_sum: int
) -> np.ndarray:
    reshaped_array = stacked_trials.reshape(stacked_trials.shape[0], -1, bin_sum)
    # Sum along the last axis to get the sums of pairs
    return np.sum(reshaped_array, axis=-1)


def mean_across_ripples(session: Session, plot: bool = True) -> np.ndarray:
    all_ripple_times = np.array(
        [
            event.peak_idx / session.sampling_rate_lfp
            for event in session.ripples_summary.events
            if event
        ]
    )
    mean_firing_around_ripples = np.empty((len(session.clusters_info), 200))
    for cluster in range(len(session.clusters_info)):
        mean_firing_around_ripples[cluster] = count_spikes_around_ripple(
            all_ripple_times,
            session.clusters_info[cluster].spike_times,
            padding=2,
            num_bins=200,
        )

    mean_firing_around_ripples_smoothed = smooth_ripple_triggered_average(
        mean_firing_around_ripples, 2
    )

    if plot:
        area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}
        for area in area_map:
            ind_area = get_clusters_ind_per_region(session, area)
            plt.figure()
            plt.pcolormesh(
                zscore(
                    mean_firing_around_ripples[min(ind_area) : max(ind_area)], axis=1
                )
            )
            plt.title(session.id + ", " + session.baseline + ", " + area_map[area])
            plt.show()

    return np.array(mean_firing_around_ripples_smoothed)


def get_clusters_ind_per_region(session: Session, area: str) -> List:
    cluster_region = [cluster.region for cluster in session.clusters_info]
    clusters_keep = [
        idx
        for idx, region in enumerate(cluster_region)
        if region is not None and area in region.lower()
    ]
    return clusters_keep


def plot_ripple_triggered_spikes(
    data: np.ndarray[any, any], region: str, color: str, session_id: List[str]
) -> None:

    data = mean_across_sessions(
        [
            SessionToAverage(session_id[session], data[session])
            for session in range(len(session_id))
        ]
    )

    padding_seconds = 2  # From analysis.py
    x_axis = np.linspace(-padding_seconds, padding_seconds, data.shape[1])

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

    plt.xlim(-2, 2)


def plot_ripple_triggered_firing(
    WTs: List[Session], NLGFs: List[Session], good_units_only: bool = True
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
        mean_across_ripples(session, plot=False) for session in WTs
    ]
    mean_firing_around_ripples_all_clusters_NLGFs = [
        mean_across_ripples(session, plot=False) for session in NLGFs
    ]

    area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}

    fig = plt.figure(figsize=(4 * 3, 4))
    n = 0
    for area in area_map:
        n = n + 1
        # using the nan mean because z_scoring for silent cells without firing around ripples will give back nans
        mean_per_session_WTs = np.vstack(
            [
                np.nanmean(
                    zscore(
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
                    zscore(
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
        )
        plot_ripple_triggered_spikes(
            mean_per_session_NLGFs,
            area,
            "red",
            session_id_NLGFs,
        )

        plt.ylabel("Firing Rate (z-score)")
        plt.xlabel("Time from ripple (s)")
        plt.title(area_map[area])
        fig.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
        fig.suptitle("Averaging across ripples per cluster first")


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


def mean_per_ripple(session: Session, area: str) -> np.ndarray:
    all_ripple_times = np.array(
        [
            event.peak_idx / session.sampling_rate_lfp
            for event in session.ripples_summary.events
            if event
        ]
    )
    clusters_keep = get_clusters_ind_per_region(session, area)

    # all the spike times for cluster of one region collapsed
    spike_times = np.hstack(
        [
            session.clusters_info[cluster].spike_times
            for cluster in range(len(session.clusters_info))
            if cluster in clusters_keep
        ]
    )

    mean_firing_around_ripples = np.empty((len(all_ripple_times), 200))
    for ripple in range(len(all_ripple_times)):
        ripple_peak = all_ripple_times[ripple].astype(float)
        mean_firing_around_ripples[ripple] = count_spikes_around_ripple_simple(
            ripple_peak,
            spike_times,
            padding=2,
            num_bins=200,
        )

    return mean_firing_around_ripples


def process_ripple_triggered_average_session(
    session: np.ndarray[Any, Any]
) -> np.ndarray:
    """The error in the mean needs to be considered here"""

    stacked_trials = np.vstack([s for s in session])
    stacked_trials = smooth_ripple_triggered_average(stacked_trials, 2)
    stacked_trials = zscore(stacked_trials, axis=1)

    mean_across_trials = np.mean(stacked_trials, axis=0)
    return mean_across_trials


def plot_ripple_triggered_firing_average_per_ripple_first(
    WTs: List[Session], NLGFs: List[Session], good_units_only: bool = True
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

    area_map = {"dg-": "dentate", "ca1": "ca1", "rsp": "retrosplenial"}

    fig = plt.figure(figsize=(4 * 3, 4))
    n = 0
    for area in area_map:
        n = n + 1
        plt.subplot(1, 3, n)
        plot_ripple_triggered_spikes(
            np.vstack(
                [
                    process_ripple_triggered_average_session(
                        mean_per_ripple(session, area)
                    )
                    for session in WTs
                ]
            ),
            "WT",
            "blue",
            [session.id for session in WTs],
        )
        plot_ripple_triggered_spikes(
            np.vstack(
                [
                    process_ripple_triggered_average_session(
                        mean_per_ripple(session, area)
                    )
                    for session in NLGFs
                ]
            ),
            "NLGF",
            "red",
            [session.id for session in NLGFs],
        )
        plt.ylabel("Firing Rate (z-score)")
        plt.xlabel("Time from ripple (s)")
        plt.title(area_map[area])
        fig.suptitle("Averaging across clusters per ripple first")


def filter_ripples_new(
    WTs: List[Session], NLGFs: List[Session]
) -> tuple[List[Session], List[Session]]:

    for session in range(len(WTs)):
        for n in range(len(WTs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(WTs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(WTs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(WTs[session].ripples_summary.ripple_freq_check[n])
            ) == False:
                WTs[session].ripples_summary.events[n] = []

    for session in range(len(NLGFs)):
        for n in range(len(NLGFs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(NLGFs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(NLGFs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(NLGFs[session].ripples_summary.ripple_freq_check[n])
            ) == False:
                NLGFs[session].ripples_summary.events[n] = []
    return WTs, NLGFs


def load_sessions() -> Tuple[List[Session], List[Session]]:

    results_files = Path(RESULTS_PATH).glob("*.json")
    WTs: List[Session] = []
    NLGFs: List[Session] = []

    for file in results_files:

        if (
            "6M" not in file.name
            # and "4M" not in file.name
            # and "5M" not in file.name
            or "A" not in file.name
        ):
            continue

        with open(file) as f:
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

    WTs, NLGFs = load_sessions()
    filter_ripples(WTs, NLGFs)
    # plot_noise_levels(WTs, NLGFs)

    # plot_ripple_characteristics(WTs, NLGFs)
    # plot_ripple_triggered_firing_average_per_ripple_first(
    #     WTs, NLGFs, good_units_only=True
    # )
    plot_ripple_triggered_firing(WTs, NLGFs, good_units_only=True)

    spikes_per_ripple(WTs, NLGFs)
    resting_time_ripple_rate_correlation(WTs, NLGFs)

    plot_single_ripples(WTs, NLGFs)

    plt.show()
