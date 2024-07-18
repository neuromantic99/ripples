from dataclasses import dataclass
from typing import List

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


class RipplesSummary(BaseModel):
    retrosplenial: List[List[int]]
    dentate: List[List[int]]
    ca1: List[List[int]]
    ripple_power: List[float]
    resting_percentage: float


@dataclass
class RotaryEncoder:
    time: np.ndarray
    position: np.ndarray
