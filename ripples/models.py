from dataclasses import dataclass
from enum import Enum
from typing import List

import numpy as np
from pydantic import BaseModel


class ClusterType(str, Enum):
    MUA = "mua"
    GOOD = "good"
    NOISE = "noise"


@dataclass
class ClusterInfo:
    spike_times: List[float]
    region: str
    info: ClusterType
    channel: int
    depth: float
    good_cluster: bool


class CandidateEvent(BaseModel):
    onset: int
    offset: int
    peak_amplitude: int | float
    peak_idx: int
    frequency: float
    bandpower_ripple: float
    detection_channel: int
    raw_lfp: List[float]


class RipplesSummary(BaseModel):
    ripple_amplitude: List[float]
    ripple_frequency: List[float]
    ripple_bandpower: List[float]
    ripple_freq_check: List[bool]
    ripple_CAR_check: List[bool]
    ripple_SRP_check: List[bool]
    ripple_CAR_check_lr: List[bool]
    ripple_SRP_check_lr: List[bool]
    resting_percentage: float
    resting_time: float
    events: List[CandidateEvent]


@dataclass
class RotaryEncoder:
    time: np.ndarray
    position: np.ndarray


class Session(BaseModel):
    ripples_summary: RipplesSummary
    clusters_info: List[ClusterInfo]
    id: str
    baseline: str
    length_seconds: float
    rms_per_channel: List[float]
    sampling_rate_lfp: float
    detection_method: str
    CA1_channels_analysed: List[int]
    CA1_channels_swr_pow: List[float]


class ProbeCoordinate(BaseModel):
    AP: float
    ML: float
    AZ: float
    elevation: float
    depth: float


@dataclass
class SessionToAverage:
    id: str
    data: float | np.ndarray
