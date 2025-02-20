from pathlib import Path
from typing import List, Tuple
import os

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import zscore

from ripples.consts import HERE, SAMPLING_RATE_LFP
from ripples.models import ClusterType, Session, SessionToAverage

from ripples.utils import mean_across_same_session, bandpass_filter, compute_envelope
from scipy import signal

sns.set_theme(context="talk", style="ticks")


RESULTS_PATH = HERE.parent / "results" / "test_1902_5Median"


def plot_ripple_raw(WTs: List[Session], NLGFs: List[Session]) -> None:

    for rec in range(len(WTs)):
        idx = range(len(WTs[rec].ripples_summary.events))

        for n in idx:
            raw = np.array(WTs[rec].ripples_summary.events[n].raw_lfp)
            raw = raw[2000:-2000]            
            if not list(raw):
                continue
            else:
                filtered = bandpass_filter(raw.reshape(1, len(raw)), 120, 250, 2500)
                envelope = signal.savgol_filter(compute_envelope(filtered), 101, 4)
                filtered = filtered.reshape(filtered.shape[1], 1)
                envelope = envelope.reshape(envelope.shape[1], 1)

                all_data = np.hstack((filtered, envelope))

                fig, axs = plt.subplots(3, 1, sharex=True)
                time = range(-500, 500, 1)

                fs = 2500
                w = 6.0
                sig = raw
                freq = np.linspace(1, fs / 2, 100)
                widths = w * fs / (2 * freq * np.pi)
                cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
                axs[0].pcolormesh(
                    time,
                    freq[0:50],
                    np.abs(cwtm[0:50, :]),
                    cmap="viridis",
                    shading="gouraud",
                )
                axs[0].axhline(80, color="red", linestyle="-")
                axs[0].axhline(250, color="red", linestyle="-")
                font = {"size": 10}
                axs[0].set_title(
                    f"freq_check: {WTs[rec].ripples_summary.ripple_freq_check[n]};CAR_check:{WTs[rec].ripples_summary.ripple_CAR_check[n]};SRP_check:{WTs[rec].ripples_summary.ripple_SRP_check[n]};CAR_check_lr:{WTs[rec].ripples_summary.ripple_CAR_check_lr[n]};SRP_check_lr:{WTs[rec].ripples_summary.ripple_SRP_check_lr[n]}",
                    fontdict=font,
                )

                onset_time = WTs[rec].ripples_summary.events[n].onset
                offset_time = WTs[rec].ripples_summary.events[n].offset
                peak_time = WTs[rec].ripples_summary.events[n].peak_idx

                # Plot each graph, and manually set the y tick values
                axs[1].plot(time, raw, linewidth=0.5)
                axs[2].plot(time, all_data, linewidth=0.5)
                axs[2].axvline((onset_time - peak_time), color="red", linestyle="-")
                axs[2].axvline(
                    (offset_time - peak_time), color="red", linestyle="-"
                )
                axs[2].set_xlabel("time (ms)")
                axs[2].set_title(
                    f"onset: {onset_time/SAMPLING_RATE_LFP}", fontdict=font
                )

                figure_path = (
                    RESULTS_PATH
                    / "figures"
                    / "ripples"
                    / f"{WTs[rec].id}"
                    / f"{rec}"
                )
                if not figure_path.exists():
                    os.makedirs(figure_path)

                plt.savefig(figure_path / f"ripple_{n}.png")
                plt.close(fig)

                ripple = raw[
                    (500 + onset_time - peak_time) : (500 + offset_time - peak_time)
                ]
                [f, Pxx] = signal.periodogram(ripple, SAMPLING_RATE_LFP)
                max_idx = np.argmax(Pxx.reshape(len(f), 1).tolist())
                max_freq = f[max_idx]
                max_val = (Pxx.reshape(len(f), 1)).tolist()[max_idx]
                max_val = max_val[0]
                ev_freq = WTs[rec].ripples_summary.events[n].frequency
                peaks, prominence = signal.find_peaks(Pxx, height=0.25 * max_val)
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
                    / f"{WTs[rec].id}"
                    / f"{rec}"
                )
                if not figure_path.exists():
                    os.makedirs(figure_path)

                plt.savefig(figure_path / f"ripple_{n}.png")
                plt.close()

    for rec in range(len(NLGFs)):
        idx = range(len(NLGFs[rec].ripples_summary.events))
        for n in idx:
            raw = np.array(NLGFs[rec].ripples_summary.events[n].raw_lfp)
            raw = raw[2000:-2000]
            if not list(raw):
                continue
            else:
                filtered = bandpass_filter(raw.reshape(1, len(raw)), 120, 250, 2500)
                envelope = signal.savgol_filter(compute_envelope(filtered), 101, 4)
                filtered = filtered.reshape(filtered.shape[1], 1)
                envelope = envelope.reshape(envelope.shape[1], 1)

                all_data = np.hstack((filtered, envelope))

                fig, axs = plt.subplots(3, 1, sharex=True)
                time = range(-500, 500, 1)

                fs = 2500
                w = 6.0
                sig = raw
                freq = np.linspace(1, fs / 2, 100)
                widths = w * fs / (2 * freq * np.pi)
                cwtm = signal.cwt(sig, signal.morlet2, widths, w=w)
                axs[0].pcolormesh(
                    time,
                    freq[0:50],
                    np.abs(cwtm[0:50, :]),
                    cmap="viridis",
                    shading="gouraud",
                )
                axs[0].axhline(80, color="red", linestyle="-")
                axs[0].axhline(250, color="red", linestyle="-")
                font = {"size": 10}
                axs[0].set_title(
                    f"freq_check: {NLGFs[rec].ripples_summary.ripple_freq_check[n]};CAR_check:{NLGFs[rec].ripples_summary.ripple_CAR_check[n]};SRP_check:{NLGFs[rec].ripples_summary.ripple_SRP_check[n]};CAR_check_lr:{NLGFs[rec].ripples_summary.ripple_CAR_check_lr[n]};SRP_check_lr:{NLGFs[rec].ripples_summary.ripple_SRP_check_lr[n]}",
                    fontdict=font,
                )

                onset_time = NLGFs[rec].ripples_summary.events[n].onset
                offset_time = NLGFs[rec].ripples_summary.events[n].offset
                peak_time = NLGFs[rec].ripples_summary.events[n].peak_idx

                # Plot each graph, and manually set the y tick values
                axs[1].plot(time, raw, linewidth=0.5)
                axs[2].plot(time, all_data, linewidth=0.5)
                axs[2].axvline((onset_time - peak_time), color="red", linestyle="-")
                axs[2].axvline(
                    (offset_time - peak_time), color="red", linestyle="-"
                )
                axs[2].set_xlabel("time (ms)")
                axs[2].set_title(
                    f"onset: {onset_time/SAMPLING_RATE_LFP}", fontdict=font
                )

                figure_path = (
                    HERE.parent
                    RESULTS_PATH
                    / "figures"
                    / "ripples"
                    / f"{NLGFs[rec].id}"
                    / f"{rec}"
                )
                if not figure_path.exists():
                    os.makedirs(figure_path)

                plt.savefig(figure_path / f"ripple_{n}.png")
                plt.close(fig)

                ripple = raw[
                    (500 + onset_time - peak_time) : (500 + offset_time - peak_time)
                ]
                [f, Pxx] = signal.periodogram(ripple, SAMPLING_RATE_LFP)
                max_idx = np.argmax(Pxx.reshape(len(f), 1).tolist())
                max_freq = f[max_idx]
                max_val = (Pxx.reshape(len(f), 1)).tolist()[max_idx]
                max_val = max_val[0]
                ev_freq = NLGFs[rec].ripples_summary.events[n].frequency
                peaks, prominence = signal.find_peaks(Pxx, height=0.25 * max_val)
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
                    / f"{NLGFs[rec].id}"
                    / f"{rec}"
                )
                if not figure_path.exists():
                    os.makedirs(figure_path)

                plt.savefig(figure_path / f"ripple_{n}.png")
                plt.close()


def number_of_spikes_per_cell_per_ripple(session: Session) -> float:
    """Follows the Tonegawa approach: pubmed.ncbi.nlm.nih.gov/24139046/"""
    all_ripple_times = np.array(
        [event.peak_idx / SAMPLING_RATE_LFP for event in session.ripples_summary.events if event]
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
                remove_month_from_session_id(session.id),
                number_of_spikes_per_cell_per_ripple(session),
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
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
    plt.savefig(HERE.parent / "figures" / "spkes_per_ripple.png")
    plt.show()


def get_ripple_rate(session: Session) -> float:
    resting_seconds = (
        session.length_seconds * session.ripples_summary.resting_percentage
    )

    return len([event.peak_idx for event in session.ripples_summary.events if event]) / resting_seconds


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
    plt.savefig(HERE.parent / "figures" / "resting_time_ripple_rate_correlation.png")


def time_spent_resting_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    plt.figure()
    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                session.length_seconds * session.ripples_summary.resting_percentage,
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                session.length_seconds * session.ripples_summary.resting_percentage,
            )
            for session in NLGFs
        ]
    )

    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data}, color="black")
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data})
    plt.ylabel("Time spent resting (s)")
    plt.tight_layout()
    plt.ylim(300, 600)
    plt.savefig(HERE.parent / "figures" / "time_spent_resting_plot.png")


def remove_month_from_session_id(session_id: str) -> str:
    return session_id
    return "_".join(session_id.split("_")[:3])


def number_of_ripples_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    plt.figure()
    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id), get_ripple_rate(session)
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id), get_ripple_rate(session)
            )
            for session in NLGFs
        ]
    )

    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data}, color="black")
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data})
    plt.ylabel("Resting ripple rate (Hz)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_rate.png")


# TODO: add in statistics


def ripple_amplitude_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.peak_amplitude for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in WTs
        ]
    )
    

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.peak_amplitude for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in NLGFs
        ]
    )

    plt.figure()
    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data})
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data}, showfliers=False)
    plt.ylim(0, 40)
    plt.ylabel(r"Ripple Amplitude ( $\mu$V)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_amplitude.png")


def ripple_freq_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.frequency for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.frequency for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in NLGFs
        ]
    )

    plt.figure()
    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data})
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data}, showfliers=False)
    plt.ylim(0, 250)
    plt.ylabel(r"Ripple frequency ( $\mu$V)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_frequency.png")


def ripple_bandpower_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    wt_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.bandpower_ripple for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in WTs
        ]
    )

    nlgf_data = mean_across_same_session(
        [
            SessionToAverage(
                remove_month_from_session_id(session.id),
                np.mean([event.bandpower_ripple for event in session.ripples_summary.events if event]).astype(float),
            )
            for session in NLGFs
        ]
    )

    plt.figure()
    sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data})
    sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data}, showfliers=False)
    plt.ylim(0, 250)
    plt.ylabel(r"Ripple bandpower ( $\mu$V)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_bandpower.png")


def smooth_ripple_triggered_average(
    stacked_trials: np.ndarray, bin_sum: int
) -> np.ndarray:
    reshaped_array = stacked_trials.reshape(stacked_trials.shape[0], -1, bin_sum)
    # Sum along the last axis to get the sums of pairs
    return np.sum(reshaped_array, axis=-1)


def process_ripple_triggered_average_session(session: List[List[int]]) -> np.ndarray:
    """The error in the mean needs to be considered here"""
    stacked_trials = np.vstack([s for s in session if s])

    # Removed for now for simplicity
    stacked_trials = smooth_ripple_triggered_average(stacked_trials, 2)

    mean_across_trials = np.mean(stacked_trials, axis=0)
    return zscore(mean_across_trials)


def plot_ripple_triggered_spikes(
    data: List[List[List[int]]], region: str, color: str, session_id: List[str]
) -> None:
    """Data is a List of sessions with a list of trials with a list of spike times. Quite a crazy datatype"""

    means = np.vstack(
        [process_ripple_triggered_average_session(session) for session in data]
    )

    padding_seconds = 2  # From analysis.py
    x_axis = np.linspace(-padding_seconds, padding_seconds, means.shape[1])
    # colors = cm.tab20(np.linspace(0, 1, len(data)))
    # for idx, session in enumerate(means):
    #     plt.plot(
    #         x_axis,
    #         session,
    #         color=colors[idx],
    #         label=f"{session_id[idx]} {region}",
    #         linestyle="dashed",
    #     )
    plt.plot(
        x_axis,
        np.mean(means, axis=0),
        color=color,
        label=region,
    )

    plt.fill_between(
        x_axis,
        np.mean(means, axis=0) - np.std(means, axis=0) / np.sqrt(means.shape[0]),
        np.mean(means, axis=0) + np.std(means, axis=0) / np.sqrt(means.shape[0]),
        color=color,
        alpha=0.3,
    )

    plt.xlim(-1.5, 1.5)


def plot_grand_ripple_triggered_average(
    WTs: List[Session], NLGFs: List[Session]
) -> None:

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 3, 1)
    plot_ripple_triggered_spikes(
        [session.ripples_summary.ca1 for session in WTs],
        "WT",
        "blue",
        [session.id for session in WTs],
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.ca1 for session in NLGFs],
        "NLGF",
        "red",
        [session.id for session in NLGFs],
    )
    plt.ylabel("Firing Rate (z-score)")
    plt.xlabel("Time from ripple (s)")
    plt.title("CA1")

    plt.subplot(1, 3, 2)
    plot_ripple_triggered_spikes(
        [session.ripples_summary.dentate for session in WTs],
        "WT",
        "blue",
        [session.id for session in WTs],
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.dentate for session in NLGFs],
        "NLGF",
        "red",
        [session.id for session in NLGFs],
    )
    plt.title("Denate gyrus")
    plt.xlabel("Time from ripple (s)")

    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1))
    plt.subplot(1, 3, 3)
    plot_ripple_triggered_spikes(
        [session.ripples_summary.retrosplenial for session in WTs],
        "WT",
        "blue",
        [session.id for session in WTs],
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.retrosplenial for session in NLGFs],
        "NLGF",
        "red",
        [session.id for session in NLGFs],
    )
    plt.xlabel("Time from ripple (s)")
    plt.title("Retrosplenial cortex")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "ripple_triggered_spikes.png")

def filter_ripples(
    WTs: List[Session], NLGFs: List[Session]
) -> tuple[List[Session], List[Session]]:
    
    for session in range(len(WTs)):
        for n in range(len(WTs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(WTs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(WTs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(WTs[session].ripples_summary.ripple_freq_check[n])
            ) == False:       
                WTs[session].ripples_summary.events[n]= []
                WTs[session].ripples_summary.dentate[n] = []
                WTs[session].ripples_summary.ca1[n] = []
                WTs[session].ripples_summary.retrosplenial[n] = []
                
    for session in range(len(NLGFs)):
        for n in range(len(NLGFs[session].ripples_summary.ripple_SRP_check)):
            if (
                np.array(NLGFs[session].ripples_summary.ripple_SRP_check[n])
                * np.array(NLGFs[session].ripples_summary.ripple_CAR_check_lr[n])
                * np.array(NLGFs[session].ripples_summary.ripple_freq_check[n])
            ) == False:         
                NLGFs[session].ripples_summary.events[n]= []
                NLGFs[session].ripples_summary.dentate[n] = []
                NLGFs[session].ripples_summary.ca1[n] = []
                NLGFs[session].ripples_summary.retrosplenial[n] = [] 
    return WTs,NLGFs

def load_sessions() -> Tuple[List[Session], List[Session]]:

    results_files = Path(RESULTS_PATH).glob("*.json")
    WTs: List[Session] = []
    NLGFs: List[Session] = []

    for file in results_files:

        if (
            "3M" not in file.name
            and "4M" not in file.name
            and "5M" not in file.name
            and "A" not in file.name
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
    #plot_ripple_raw(WTs, NLGFs)
    
    #plot_grand_ripple_triggered_average(WTs, NLGFs)

    spikes_per_ripple(WTs, NLGFs)
    number_of_ripples_plot(WTs, NLGFs)

    #ripple_amplitude_plot(WTs, NLGFs)
    #ripple_freq_plot(WTs, NLGFs)
    #ripple_bandpower_plot(WTs, NLGFs)
    #plot_grand_ripple_triggered_average(WTs, NLGFs)
    #time_spent_resting_plot(WTs, NLGFs)
    #resting_time_ripple_rate_correlation(WTs, NLGFs)
    plt.show()

