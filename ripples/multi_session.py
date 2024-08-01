from typing import List
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import zscore

from ripples.consts import HERE
from ripples.models import RipplesSummary, Session, Session
import seaborn as sns

sns.set_theme(context="talk", style="ticks")


RESULTS_PATH = HERE.parent / "results"


def get_ripple_rate(ripples_summary: RipplesSummary) -> float:

    # Assumes recording 10 minutes long. Change me
    resting_seconds = (10 * 60) * ripples_summary.resting_percentage

    return len(ripples_summary.ripple_power) / resting_seconds


def number_of_ripples_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    plt.figure()
    sns.boxplot(
        {
            "WTs": [get_ripple_rate(session.ripples_summary) for session in WTs],
            "NLGFs": [get_ripple_rate(session.ripples_summary) for session in NLGFs],
        }
    )
    plt.ylabel("Resting ripple rate (Hz)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_rate.png")


def ripple_power_plot(WTs: List[Session], NLGFs: List[Session]) -> None:

    plt.figure()
    sns.boxplot(
        {
            "WTs": (np.mean(session.ripples_summary.ripple_power) for session in WTs),
            "NLGFs": (
                np.mean(session.ripples_summary.ripple_power) for session in NLGFs
            ),
        }
    )
    plt.ylim(0, 40)
    plt.ylabel(r"Ripple power ( $\mu$V)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_power.png")


def smooth_ripple_triggered_average(
    stacked_trials: np.ndarray, bin_sum: int
) -> np.ndarray:
    reshaped_array = stacked_trials.reshape(stacked_trials.shape[0], -1, bin_sum)
    # Sum along the last axis to get the sums of pairs
    return np.sum(reshaped_array, axis=-1)


def process_ripple_triggered_average_session(session: List[List[int]]) -> np.ndarray:
    """The error in the mean needs to be considered here"""
    stacked_trials = np.vstack(session)

    # Removed for now for simplicity
    stacked_trials = smooth_ripple_triggered_average(stacked_trials, 2)

    mean_across_trials = np.mean(stacked_trials, axis=0)
    return zscore(mean_across_trials)


def plot_ripple_triggered_spikes(
    data: List[List[List[int]]], region: str, color: str
) -> None:
    """Data is a List of sessions with a list of trials with a list of spike times. Quite a crazy datatype"""

    means = np.vstack(
        [process_ripple_triggered_average_session(session) for session in data]
    )

    padding_seconds = 2  # From analysis.py
    plt.plot(
        np.linspace(-padding_seconds, padding_seconds, means.shape[1]),
        np.mean(means, axis=0),
        color=color,
        label=region,
    )

    plt.fill_between(
        np.linspace(-padding_seconds, padding_seconds, means.shape[1]),
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
        [session.ripples_summary.ca1 for session in WTs], "WT", "blue"
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.ca1 for session in NLGFs], "NLGF", "red"
    )
    plt.ylabel("Firing Rate (z-score)")
    plt.xlabel("Time from ripple (s)")
    plt.title("CA1")
    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1))

    plt.subplot(1, 3, 2)
    plot_ripple_triggered_spikes(
        [session.ripples_summary.dentate for session in WTs], "WT", "blue"
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.dentate for session in NLGFs], "NLGF", "red"
    )
    plt.title("Denate gyrus")
    plt.xlabel("Time from ripple (s)")

    plt.subplot(1, 3, 3)
    plot_ripple_triggered_spikes(
        [session.ripples_summary.retrosplenial for session in WTs], "WT", "blue"
    )
    plot_ripple_triggered_spikes(
        [session.ripples_summary.retrosplenial for session in NLGFs], "NLGF", "red"
    )
    plt.xlabel("Time from ripple (s)")
    plt.title("Retrosplenial cortex")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "ripple_triggered_spikes.png")

    plt.show()


def main() -> None:

    results_files = Path(RESULTS_PATH).glob("*.json")
    WTs: List[Session] = []
    NLGFs: List[Session] = []

    for file in results_files:

        if "3M" not in file.name and "4M" not in file.name:
            continue

        with open(file) as f:
            result = Session.model_validate_json(f.read())

        if "wt" in file.name.lower():
            WTs.append(result)
        elif "nlgf" in file.name.lower():
            NLGFs.append(result)
        else:
            raise ValueError(f"Unknown type of recording: {file.name}")

    number_of_ripples_plot(WTs, NLGFs)
    ripple_power_plot(WTs, NLGFs)
    plot_grand_ripple_triggered_average(WTs, NLGFs)
