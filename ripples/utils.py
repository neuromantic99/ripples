from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import signal


@dataclass
class CandidateEvent:
    onset: int
    offset: int


def detect_ripple_events(
    channel: np.ndarray | List[int | float],
) -> List[CandidateEvent]:
    median = np.median(channel)
    upper_threshold = median * 5
    lower_threshold = median * 2.5
    candidate_events: List[CandidateEvent] = []

    in_event = False
    upper_exceeded = False
    start_event = 0

    for idx, value in enumerate(channel):
        if value > lower_threshold and not in_event:
            start_event = idx
            in_event = True

        # If you bounce on the lower threshold
        if value < lower_threshold and in_event and not upper_exceeded:
            in_event = False

        if value > upper_threshold:
            upper_exceeded = True

        if value < lower_threshold and in_event and upper_exceeded:
            in_event = False
            upper_exceeded = False
            candidate_events.append(CandidateEvent(onset=start_event, offset=idx))

    return candidate_events


def bandpass_filter(
    lfp: np.ndarray, low: int, high: int, sampling_rate: int
) -> np.ndarray:
    b, a = signal.butter(4, Wn=[low, high], fs=sampling_rate, btype="bandpass")
    return signal.filtfilt(b, a, lfp, axis=1)
