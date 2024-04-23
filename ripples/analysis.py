from typing import List
import numpy as np
from scipy import signal

from utils import CandidateEvent, bandpass_filter, detect_ripple_events

from utils_npyx import load_lfp

from consts import CA1_channels

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe
DATA_PATH = "data/031123_g0_imec0"
DATA_PATH_MAT = "data/lfp_data_noCAR_fin_imec1.mat"

CA1_channels = [channel - 1 for channel in CA1_channels]

SAMPLING_RATE = 2500


def preprocess(lfp: np.ndarray) -> np.ndarray:
    """Sam also centers the data"""
    # lfp = signal.decimate(lfp, 2)
    return lfp - np.mean(lfp, 0)


def compute_envelope(lfp: np.ndarray) -> np.ndarray:
    hilbert_transformed = signal.hilbert(lfp, axis=1)
    return np.abs(hilbert_transformed)


def length_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    return [event for event in candidate_events if event.offset - event.onset > 41]


if __name__ == "__main__":
    lfp = load_lfp(DATA_PATH)
    lfp[REFERENCE_CHANNEL, :] = 0

    # The long linear channels are interleaved
    lfp = np.concatenate((lfp[0::2, :], lfp[1::2, :]), axis=0)
    lfp = lfp[CA1_channels, :]

    # Trim to match Suraya, remove in future
    lfp = lfp[:, 6589:1537452]

    lfp = preprocess(lfp)
    lfp = bandpass_filter(lfp, 125, 250, SAMPLING_RATE)
    lfp = compute_envelope(lfp)

    candidate_events: List[List[CandidateEvent]] = []
    for channel in lfp:
        candidate_events.append(detect_ripple_events(channel))

    candidate_events = [length_check(events) for events in candidate_events]

    num_events = sum(len(events) for events in candidate_events)

    1 / 0
