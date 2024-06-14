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
