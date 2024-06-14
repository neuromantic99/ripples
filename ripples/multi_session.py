from matplotlib import pyplot as plt
import numpy as np
from pydantic import BaseModel

from ripples.models import Result


def main():
    with open(
        "/Users/jamesrowland/Code/ripples/results/WT_A_1397747_3M-baseline1-1.json"
    ) as f:
        result = Result.model_validate_json(f.read())

    plt.plot(np.mean(result.retrosplenial, axis=0), color="red")
    plt.plot(np.mean(result.dentate, axis=0), color="blue")
    plt.plot(np.mean(result.ca1, axis=0), color="black")
    plt.show()

    # plt.plot(np.mean(spike_count, axis=0))
    # # Make this an odd number

    # n_ticks = 11
    # # This might be off by one, so be very careful if doing super precise alignment to 0
    # plt.xticks(
    #     np.linspace(0, n_bins, n_ticks),
    #     np.round(np.linspace(-padding, padding, n_ticks), 1),
    # )

    # plt.axvline(spike_count.shape[1] / 2, color="black", linestyle="--")
    # plt.xlabel("Time from ripple (s)")
