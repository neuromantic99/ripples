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


class Session(BaseModel):
    ripples_summary: RipplesSummary
    clusters_info: List[ClusterInfo]


class ProbeCoordinate(BaseModel):
    AP: float
    ML: float
    AZ: float
    elevation: float
    depth: float
