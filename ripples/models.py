from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, computed_field
from ripples.consts import SAMPLING_RATE_LFP


@dataclass
class SpikesSession:
    spike_channels: np.ndarray
    spike_times: np.ndarray


class CandidateEvent(BaseModel):
    onset: int
    offset: int
    peak_power: int | float
    peak_idx: int

    @computed_field
    @property
    def peak_time(self) -> float:
        return self.peak_idx / SAMPLING_RATE_LFP
