from typing import List
import numpy as np

from ripples.utils import CandidateEvent, bandpass_filter, compute_envelope


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
) -> np.ndarray:

    spike_times = spike_times[
        np.logical_and(
            spike_times > (ripple.peak_time - padding),
            spike_times < (ripple.peak_time + padding),
        )
    ]

    counts, _ = np.histogram(spike_times, bins=num_bins)
    return counts


def remove_duplicate_ripples(
    ripples: List[CandidateEvent], min_distance_seconds: float
) -> List[CandidateEvent]:
    """TODO: This is very ineffecient."""

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
