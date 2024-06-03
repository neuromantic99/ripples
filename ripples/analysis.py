from typing import List
import numpy as np
from scipy import signal
from pathlib import Path

from gsheets_importer import gsheet2df
from utils import CandidateEvent, bandpass_filter, detect_ripple_events

from utils_npyx import load_lfp

from consts import aligned_regions

import mat73

REFERENCE_CHANNEL = 191  # For the long linear, change depending on probe


UMBRELLA = Path("/Users/jamesrowland/Documents/data/")
SESSION = "NLGF_A_1393311_3M"
RECORDING_NAME = "baseline1"
PROBE = "1"


# TODO: Get this from the probe details
CA1_channels = [
    348 - idx for idx, region in enumerate(aligned_regions) if region == "Field CA1"
]


SAMPLING_RATE = 2500


def preprocess(lfp: np.ndarray) -> np.ndarray:
    """Center and decimate, check what preprocessing is necessary"""
    # lfp = signal.decimate(lfp, 2)
    return lfp - np.mean(lfp, 0)


def compute_envelope(lfp: np.ndarray) -> np.ndarray:
    hilbert_transformed = signal.hilbert(lfp, axis=1)
    return np.abs(hilbert_transformed)


def length_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    return [event for event in candidate_events if event.offset - event.onset > 41]


def event_power_check(
    candidate_events: List[CandidateEvent], comparison_power: np.ndarray
):
    return [
        event
        for event in candidate_events
        if event.peak_power >= comparison_power[event.peak_idx] * 2
    ]


def get_candidate_ripples(lfp: np.ndarray) -> List[List[CandidateEvent]]:
    """Gets candidate ripples from a common average referenced LFP"""
    ripple_band = compute_envelope(bandpass_filter(lfp, 125, 250, SAMPLING_RATE))
    return [detect_ripple_events(channel) for channel in ripple_band]


def filter_candidate_ripples(
    candidate_events: List[List[CandidateEvent]],
    lfp: np.ndarray,
    lfp_raw: np.ndarray,
) -> List[List[CandidateEvent]]:

    candidate_events = [length_check(events) for events in candidate_events]

    common_average_power = compute_envelope(
        bandpass_filter(
            np.expand_dims(np.mean(lfp_raw, 0), axis=0), 125, 250, SAMPLING_RATE
        )
    )

    candidate_events = [
        event_power_check(events, common_average_power.squeeze())
        for events in candidate_events
    ]

    supra_ripple_band_power = compute_envelope(
        bandpass_filter(lfp, 200, 500, SAMPLING_RATE)
    )

    return [
        event_power_check(events, supra)
        for events, supra in zip(candidate_events, supra_ripple_band_power)
    ]


if __name__ == "__main__":

    metadata = gsheet2df("1HSERPbm-kDhe6X8bgflxvTuK24AfdrZJzbdBy11Hpcg", "Sheet1", 1)

    metadata_probe = metadata[
        (metadata["Session"] == SESSION)
        & (metadata["Recording Name"] == RECORDING_NAME)
        & (metadata["Probe"] == PROBE)
    ]
    lfp_path = metadata_probe["LFP path"].values[0]

    probe_details_path = metadata_probe["Probe Details Path"].values[0]

    lfp_raw = load_lfp(UMBRELLA / lfp_path)

    probe_details = mat73.loadmat(UMBRELLA / probe_details_path)

    lfp_raw[REFERENCE_CHANNEL, :] = 0

    # The long linear channels are interleaved
    lfp_raw = np.concatenate((lfp_raw[0::2, :], lfp_raw[1::2, :]), axis=0)
    lfp_raw = lfp_raw[CA1_channels, :]

    # Common average reference
    lfp = preprocess(lfp_raw)
    candidate_events = get_candidate_ripples(lfp)
    ripples = filter_candidate_ripples(candidate_events, lfp, lfp_raw)

    num_events = sum(len(events) for events in ripples)

    1 / 0
