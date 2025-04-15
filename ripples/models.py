from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Any

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
    MFR_resting: float | str
    Firing_rate_modulation: float | str
    cell_type: float | str
    ripple_modulation: float | str
    wf_max_channel: int
    waveform_mean_norm: List
    aligned_wf: List
    valley_to_peak_time: float
    halfwidth_at_third_max: float


class CandidateEvent(BaseModel):
    onset: int
    offset: int
    peak_amplitude: int | float
    peak_idx: int
    frequency: float
    bandpower_ripple: float
    strength: float
    detection_channel: int
    instantaneous_frequency: float
    raw_lfp: List[float]


class RipplesSummary(BaseModel):
    ripple_amplitude: List[float]
    ripple_frequency: List[float]
    ripple_bandpower: List[float]
    ripple_strength: List[float]
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
    unique_id: str
    length_seconds: float
    rms_per_channel: List[float]
    sampling_rate_lfp: float
    detection_method: str
    CA1_channels_analysed: List[int]
    CA1_channels_swr_pow: List[float]
    resting_ind: List[bool]
    resting_ind_strict: List[bool]
    resting_periods_ind: List[List[int]]
    locomotion_periods_ind: List[List[int]]
    resting_time_strict: int
    locomotion_time_strict: int


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
