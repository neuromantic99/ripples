from dataclasses import dataclass

import numpy as np


@dataclass
class SpikesSession:
    spike_channels: np.ndarray
    spike_times: np.ndarray


@dataclass
class CandidateEvent:
    onset: int
    offset: int
    peak_power: int | float
    peak_idx: int
    # peak_time: float
