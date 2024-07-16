from typing import List
import numpy as np

from ripples.models import CandidateEvent, RotaryEncoder
from ripples.utils import bandpass_filter, compute_envelope


def length_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    return [event for event in candidate_events if event.offset - event.onset > 41]


def event_power_check(
    candidate_events: List[CandidateEvent], comparison_power: np.ndarray
) -> List[CandidateEvent]:
    return [
        event
        for event in candidate_events
        if event.peak_power >= comparison_power[event.peak_idx] * 2
    ]


def filter_candidate_ripples(
    candidate_events: List[List[CandidateEvent]],
    lfp: np.ndarray,
    common_average: np.ndarray,
    sampling_rate: int,
) -> List[List[CandidateEvent]]:

    assert (
        len(candidate_events) == lfp.shape[0]
        and lfp.shape[1] == common_average.shape[0]
    )

    candidate_events = [length_check(events) for events in candidate_events]

    common_average_power = compute_envelope(
        bandpass_filter(np.expand_dims(common_average, axis=0), 125, 250, sampling_rate)
    )

    candidate_events = [
        event_power_check(events, common_average_power.squeeze())
        for events in candidate_events
    ]

    supra_ripple_band_power = compute_envelope(
        bandpass_filter(lfp, 200, 500, sampling_rate)
    )

    return [
        event_power_check(events, supra)
        for events, supra in zip(candidate_events, supra_ripple_band_power)
    ]


def count_spikes_around_ripple(
    ripple: CandidateEvent, spike_times: np.ndarray, padding: float, num_bins: int
) -> List[float]:

    spike_times = spike_times[
        np.logical_and(
            spike_times > (ripple.peak_time - padding),
            spike_times < (ripple.peak_time + padding),
        )
    ]

    counts, _ = np.histogram(spike_times, bins=num_bins)
    return list(counts.astype(float))


def remove_duplicate_ripples(
    ripples: List[CandidateEvent], min_distance_seconds: float
) -> List[CandidateEvent]:
    """TODO: This is very ineffecient and will break if peak power is exactly the same across two ripples"""

    filtered_ripples: List[CandidateEvent] = []

    for i in range(len(ripples)):
        keep = True
        for j in range(len(ripples)):
            if i == j:
                continue
            if (
                abs(ripples[i].peak_time - ripples[j].peak_time) < min_distance_seconds
                and ripples[i].peak_power < ripples[j].peak_power
            ):
                keep = False
                break
        if keep:
            filtered_ripples.append(ripples[i])

    return filtered_ripples


def get_resting_ripples(
    ripples: List[CandidateEvent], rotary_encoder: RotaryEncoder, threshold: float
) -> List[CandidateEvent]:
    """Double check this is 100% correct. I also don't love the interpolation when it only samples during actual movement"""
    resting_ripples = []
    for ripple in ripples:
        ripple_speed = np.interp(
            ripple.peak_time, rotary_encoder.time, rotary_encoder.speed
        )
        if abs(ripple_speed) < threshold:
            resting_ripples.append(ripple)

    return resting_ripples


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
    peak_power = -np.inf

    for idx, value in enumerate(channel):

        if value > lower_threshold and not in_event:
            start_event = idx
            in_event = True

        if in_event and value > peak_power:
            peak_power = value
            peak_idx = idx

        # If you bounce on the lower threshold
        if value < lower_threshold and in_event and not upper_exceeded:
            in_event = False

        if value > upper_threshold:
            upper_exceeded = True

        if value < lower_threshold and in_event and upper_exceeded:
            in_event = False
            upper_exceeded = False
            candidate_events.append(
                CandidateEvent(
                    onset=start_event,
                    offset=idx,
                    peak_power=peak_power,
                    peak_idx=peak_idx,
                )
            )
            peak_power = -np.inf

    return candidate_events


def get_candidate_ripples(
    lfp: np.ndarray, sampling_rate: int
) -> List[List[CandidateEvent]]:
    """Gets candidate ripples from a common average referenced LFP"""
    ripple_band = compute_envelope(bandpass_filter(lfp, 125, 250, sampling_rate))
    return [detect_ripple_events(channel) for channel in ripple_band]
