from typing import List
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import zscore

from ripples.consts import HERE
from ripples.models import Result
import seaborn as sns

sns.set_theme(context="talk", style="ticks")


RESULTS_PATH = HERE.parent / "results"

""" Everything here is in a beta phase and needs considering and testing"""


def get_ripple_rate(result: Result) -> float:

    # Assumes recording 10 minutes long. Change me
    resting_seconds = (10 * 60) * result.resting_percentage

    return len(result.ripple_power) / resting_seconds


def number_of_ripples_plot(WTs: List[Result], NLGFs: List[Result]) -> None:

    plt.figure()
    sns.boxplot(
        {
            "WTs": [get_ripple_rate(result) for result in WTs],
            "NLGFs": [get_ripple_rate(result) for result in NLGFs],
        }
    )
    plt.ylabel("Resting ripple rate (Hz)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_rate.png")


def ripple_power_plot(WTs: List[Result], NLGFs: List[Result]) -> None:

    plt.figure()
    sns.boxplot(
        {
            # "WTs": flatten([result.ripple_power for result in WTs]),
            # "NLGFs": flatten([result.ripple_power for result in NLGFs]),
            "WTs": (np.mean(result.ripple_power) for result in WTs),
            "NLGFs": (np.mean(result.ripple_power) for result in NLGFs),
        }
    )
    plt.ylim(0, 40)
    plt.ylabel(r"Ripple power ( $\mu$V)")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "resting_ripple_power.png")


def process_session(session: List[List[float]], bin_sum: int = 2) -> np.ndarray:
    stacked = np.vstack(session)
    reshaped_array = stacked.reshape(stacked.shape[0], -1, bin_sum)
    # # Sum along the last axis to get the sums of pairs
    summed_array = np.sum(reshaped_array, axis=-1)
    mean_across_trials = np.mean(summed_array, axis=0)
    return zscore(mean_across_trials)


def plot_ripple_triggered_spikes(
    data: List[List[List[float]]], region: str, color: str
) -> None:
    """This datatype is quite crazy"""

    means = np.vstack([process_session(session, 2) for session in data])

    # means = np.vstack([zscore(np.mean(np.vstack(session), axis=0)) for session in data])
    # assert means.shape[0] == len(data )

    # reshaped_array = means.reshape(means.shape[0], -1, 2)
    # # Sum along the last axis to get the sums of pairs
    # summed_array = np.sum(reshaped_array, axis=-1)

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

    # n_ticks = 5  # This might be off by one, so be very careful if doing super precise alignment to 0
    # n = means.shape[1]


def main() -> None:

    results_files = Path(RESULTS_PATH).glob("*.json")
    WTs: List[Result] = []
    NLGFs: List[Result] = []

    for file in results_files:

        if "3M" not in file.name and "4M" not in file.name:
            continue

        with open(file) as f:
            result = Result.model_validate_json(f.read())

        if "wt" in file.name.lower():
            WTs.append(result)
        elif "nlgf" in file.name.lower():
            NLGFs.append(result)
        else:
            raise ValueError(f"Unknown type of recording: {file.name}")

    # number_of_ripples_plot(WTs, NLGFs)
    # ripple_power_plot(WTs, NLGFs)

    plt.figure(figsize=(4 * 3, 4))
    plt.subplot(1, 3, 1)
    plot_ripple_triggered_spikes([result.ca1 for result in WTs], "WT", "blue")
    plot_ripple_triggered_spikes([result.ca1 for result in NLGFs], "NLGF", "red")
    plt.ylabel("Firing Rate (z-score)")
    plt.xlabel("Time from ripple (s)")
    plt.title("CA1")
    plt.legend(loc="upper right", bbox_to_anchor=(1.05, 1))

    plt.subplot(1, 3, 2)
    plot_ripple_triggered_spikes([result.dentate for result in WTs], "WT", "blue")
    plot_ripple_triggered_spikes([result.dentate for result in NLGFs], "NLGF", "red")
    plt.title("Denate gyrus")
    plt.xlabel("Time from ripple (s)")

    plt.subplot(1, 3, 3)
    plot_ripple_triggered_spikes([result.retrosplenial for result in WTs], "WT", "blue")
    plot_ripple_triggered_spikes(
        [result.retrosplenial for result in NLGFs], "NLGF", "red"
    )
    plt.xlabel("Time from ripple (s)")
    plt.title("Retrosplenial cortex")
    plt.tight_layout()
    plt.savefig(HERE.parent / "figures" / "ripple_triggered_spikes.png")

    plt.show()
