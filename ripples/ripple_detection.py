from typing import List
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from ripples.models import CandidateEvent, RotaryEncoder
from ripples.utils import (
    bandpass_filter,
    compute_envelope,
    smallest_positive_index,
    bandpower,
)


def length_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    return [event for event in candidate_events if event.offset - event.onset > 41]


def event_power_check(
    candidate_events: List[CandidateEvent], comparison_power: np.ndarray
) -> List[CandidateEvent]:
    return [
        event
        for event in candidate_events
        if event.peak_power >= comparison_power[event.peak_idx] * 1.5
    ]


def frequency_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    min_freq = 100  # 100Hz, eLife paper
    return [event for event in candidate_events if event.frequency > min_freq]


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

    ripples = [event for events in candidate_events for event in events]
    print(f"Number of ripples after length check: {len(ripples)}")

    candidate_events = [frequency_check(events) for events in candidate_events]

    ripples1 = [event for events in candidate_events for event in events]
    print(f"Number of ripples after frequency check: {len(ripples1)}")

    # common_average_power = compute_envelope(
    #     bandpass_filter(np.expand_dims(common_average, axis=0), 80, 250, sampling_rate)
    # )

    # candidate_events = [
    #     event_power_check(events, common_average_power.squeeze())
    #     for events in candidate_events
    # ]

    # ripples2 = [event for events in candidate_events for event in events]
    # print(f"Number of ripples after CAR check: {len(ripples2)}")

    # supra_ripple_band_power = compute_envelope(
    #     bandpass_filter(lfp, 200, 500, sampling_rate)
    # )

    #     event_power_check(events, supra)
    # for events, supra in zip(candidate_events, supra_ripple_band_power)

    return candidate_events


def count_spikes_around_ripple(
    ripple: CandidateEvent,
    spike_times: np.ndarray,
    padding: float,
    num_bins: int,
    sampling_rate_lfp: int,
) -> List[float]:

    peak_time = ripple.peak_idx / sampling_rate_lfp

    spike_times = spike_times[
        np.logical_and(
            spike_times > (peak_time - padding),
            spike_times < (peak_time + padding),
        )
    ]

    counts, _ = np.histogram(spike_times, bins=num_bins)
    return list(counts.astype(float))

def remove_duplicate_ripples(
    ripples: List[CandidateEvent], min_distance_seconds: float, sampling_rate_lfp: int
) -> List[CandidateEvent]:
    """TODO: This is very ineffecient and will break if peak power is exactly the same across two ripples"""

    filtered_ripples: List[CandidateEvent] = []
    for i in range(len(ripples)):
        keep = True
        for j in range(len(ripples)):
            if i == j:
                continue
            if (
                abs(
                    ripples[i].peak_idx / sampling_rate_lfp
                    - ripples[j].peak_idx / sampling_rate_lfp
                )
                < min_distance_seconds
                and ripples[i].peak_power < ripples[j].peak_power
            ):
                keep = False
                break
        if keep:
            filtered_ripples.append(ripples[i])

    return filtered_ripples


def average_ripple_speed(
    ripple: CandidateEvent, rotary_encoder: RotaryEncoder, sampling_rate_lfp: int
) -> float:
    """This assumes the mouse is running in one direction only which is probably not correct"""

    start_time = ripple.onset / sampling_rate_lfp
    end_time = ripple.offset / sampling_rate_lfp
    start_idx = smallest_positive_index(start_time - rotary_encoder.time)
    end_idx = smallest_positive_index(end_time - rotary_encoder.time)
    distance = rotary_encoder.position[end_idx] - rotary_encoder.position[start_idx]
    return distance / (end_time - start_time)


def rotary_encoder_percentage_resting(
    rotary_encoder: RotaryEncoder, threshold: float, max_time: float, plot: bool = False
) -> Tuple[float, float]:
    """Checked with plotting but writ tests"""

    bin_size = 1
    bin_edges = np.arange(0, max_time, bin_size)

    speed = []
    for idx in range(len(bin_edges) - 1):
        start_time = bin_edges[idx]
        end_time = bin_edges[idx + 1]
        start_idx = smallest_positive_index(start_time - rotary_encoder.time)
        end_idx = smallest_positive_index(end_time - rotary_encoder.time)
        distance = rotary_encoder.position[end_idx] - rotary_encoder.position[start_idx]
        speed.append(distance / (end_time - start_time))

    speed = np.array(speed)
    resting_percentage = sum(speed < threshold) / len(speed)
    resting_time = sum(speed < threshold)

    if plot:
        _, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(bin_edges[:-1], speed, color="red")
        ax2.plot(rotary_encoder.time, rotary_encoder.position)

    return [resting_percentage, resting_time]


def get_resting_ripples(
    ripples: List[CandidateEvent],
    rotary_encoder: RotaryEncoder,
    threshold: float,
    sampling_rate_lfp: int,
) -> List[CandidateEvent]:
    """Assumes that the animal has not run forwards and backwards during the ripple"""
    return [
        ripple
        for ripple in ripples
        if (
            abs(average_ripple_speed(ripple, rotary_encoder, sampling_rate_lfp))
            < threshold
        )
    ]


def detect_ripple_events(
    channel: int, lfp: np.ndarray, CA1_channels: List[int], sampling_rate: int
) -> List[CandidateEvent]:

    print(lfp[channel, :].size)
    ripple_band_pre = compute_envelope(
        bandpass_filter(
            lfp[channel, :].reshape(1, lfp[channel, :].size), 80, 250, sampling_rate
        )
    )  # freq range Dupret [80 250]
    ripple_band = signal.savgol_filter(ripple_band_pre, 101, 4)
    ripple_band = ripple_band.reshape(lfp[channel, :].size, 1)
    median = np.median(ripple_band)
    upper_threshold = (
        median * 5
    )  # see if change to 5*SD makes sense (as done in elife paper)
    lower_threshold = median * 2.5
    candidate_events: List[CandidateEvent] = []

    in_event = False
    upper_exceeded = False
    start_event = 0
    peak_power = -np.inf

    for idx, value in enumerate(ripple_band):

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
         
            [f, Pxx] = signal.periodogram(lfp[0, start_event:idx], fs=sampling_rate)
            Max = np.argmax(Pxx.reshape(len(f), 1).tolist())
            Max_freq = f[Max]
            Max_val = (Pxx.reshape(len(f), 1)).tolist()[Max]
            Max_val = Max_val[0]

        
            bandpower_ripple = bandpower(
                lfp[0, start_event:idx], sampling_rate, 80, 250
            )

            candidate_events.append(
                CandidateEvent(
                    onset=start_event,
                    offset=idx,
                    peak_power=peak_power,
                    peak_idx=peak_idx,
                    detection_channel=CA1_channels[channel],
                    frequency=Max_freq,
                    bandpower_ripple=bandpower_ripple,
                )
            )
            peak_power = -np.inf
            Max_freq = -np.inf
            bandpower_ripple = -np.inf

    return candidate_events


def get_candidate_ripples(
    lfp: np.ndarray,
    CA1_channels: List[int],
    sampling_rate: int,
) -> List[List[CandidateEvent]]:
    """Gets candidate ripples from a common average referenced LFP"""
    channel_idx = list(range(0, len(CA1_channels)))
    return [
        detect_ripple_events(channel, lfp, CA1_channels, sampling_rate)
        for channel in channel_idx
    ]
