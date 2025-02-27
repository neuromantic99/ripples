from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from ripples.models import CandidateEvent, RotaryEncoder
from ripples.utils import (
    bandpass_filter,
    compute_envelope,
    smallest_positive_index,
    bandpower,
    get_event_frequency,
)
from ripples.consts import SUPRA_RIPPLE_BAND, RIPPLE_BAND


def length_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    return [event for event in candidate_events if event.offset - event.onset > 41]


def event_power_check(
    candidate_events: List[CandidateEvent], comparison_power: np.ndarray
) -> List[CandidateEvent]:
    return [
        event
        for event in candidate_events
        if event.peak_amplitude >= comparison_power[event.peak_idx] * 2
    ]


def frequency_check(candidate_events: List[CandidateEvent]) -> List[CandidateEvent]:
    min_freq = 100  # 100Hz, eLife paper
    return [event for event in candidate_events if event.frequency > min_freq]


def get_quality_metrics(
    candidate_events: List[CandidateEvent],
    lfp: np.ndarray,
    common_average: np.ndarray,
    sampling_rate: float,
) -> Tuple[list, list, list, list, list, List[CandidateEvent]]:

    assert lfp.shape[1] == common_average.shape[0]

    candidate_events = [
        event for event in candidate_events if event.offset - event.onset > 41
    ]

    print(
        f"Number of ripples after length check: {len([event for event in candidate_events])}"
    )

    freq_check = [event.frequency > 100 for event in candidate_events]
    common_average_power = compute_envelope(
        bandpass_filter(
            np.expand_dims(common_average, axis=0),
            RIPPLE_BAND[0],
            RIPPLE_BAND[1],
            sampling_rate,
        )
    )
    common_average_power = common_average_power.reshape(common_average_power.shape[1])
    CAR_check = [
        event.peak_amplitude >= common_average_power[event.peak_idx] * 2
        for event in candidate_events
    ]
    CAR_check_lr = [
        event.peak_amplitude >= common_average_power[event.peak_idx] * 1.5
        for event in candidate_events
    ]
    supra_ripple_band_power = compute_envelope(
        bandpass_filter(
            lfp[2, :].reshape(1, lfp.shape[1]),
            SUPRA_RIPPLE_BAND[0],
            SUPRA_RIPPLE_BAND[1],
            sampling_rate,
        )
    )
    supra_ripple_band_power = supra_ripple_band_power.reshape(
        supra_ripple_band_power.shape[1]
    )
    SRP_check = [
        event.peak_amplitude >= supra_ripple_band_power[event.peak_idx] * 2
        for event in candidate_events
    ]
    SRP_check_lr = [
        event.peak_amplitude >= supra_ripple_band_power[event.peak_idx] * 1.5
        for event in candidate_events
    ]

    return (
        freq_check,
        CAR_check,
        SRP_check,
        CAR_check_lr,
        SRP_check_lr,
        candidate_events,
    )


def count_spikes_around_ripple(
    ripple: CandidateEvent,
    spike_times: np.ndarray,
    padding: float,
    num_bins: int,
    sampling_rate_lfp: float,
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
    ripples: List[CandidateEvent], min_distance_seconds: float, sampling_rate_lfp: float
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
                and ripples[i].peak_amplitude < ripples[j].peak_amplitude
            ):
                keep = False
                break
        if keep:
            filtered_ripples.append(ripples[i])

    assert len(np.unique([ripple.peak_idx for ripple in filtered_ripples])) == len(
        filtered_ripples
    )

    return filtered_ripples


def do_preprocessing_lfp_for_ripple_analysis(
    lfp: np.ndarray, sampling_rate: float, channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    ripple_band_unsmoothed = compute_envelope(
        bandpass_filter(
            lfp[channel, :].reshape(1, lfp[channel, :].size),
            RIPPLE_BAND[0],
            RIPPLE_BAND[1],
            sampling_rate,
        )
    )  # freq range Dupret [80 250]
    ripple_band = signal.savgol_filter(ripple_band_unsmoothed, 101, 4)
    ripple_band = ripple_band.reshape(lfp[channel, :].size, 1)
    ripple_band_unsmoothed = ripple_band_unsmoothed.reshape(lfp[channel, :].size, 1)
    return ripple_band, ripple_band_unsmoothed


def detect_ripple_events(
    channel: int,
    lfp_det_chans: np.ndarray,
    detection_channels_ca1: List[int],
    resting_ind: bool,
    sampling_rate: float,
    method: str,
) -> List[CandidateEvent]:

    ripple_band, ripple_band_unsmoothed = do_preprocessing_lfp_for_ripple_analysis(
        lfp_det_chans, sampling_rate, channel
    )

    assert len(ripple_band) == len(resting_ind)

    if method == "median":
        median = np.median(ripple_band[resting_ind])
        upper_threshold = median * 5
        lower_threshold = median * 2.5

    elif method == "sd":
        sd = np.std(ripple_band[resting_ind])
        upper_threshold = sd * 5
        lower_threshold = sd * 2.5

    candidate_events: List[CandidateEvent] = []

    in_event = False
    upper_exceeded = False
    start_event = 0
    peak_amp = -np.inf

    for idx, value in enumerate(ripple_band):

        if value > lower_threshold and not in_event:
            start_event = idx
            in_event = True

        if in_event and value > peak_amp:
            peak_amp = value
            peak_idx = idx

        # If you bounce on the lower threshold
        if value < lower_threshold and in_event and not upper_exceeded:
            in_event = False

        if value > upper_threshold:
            upper_exceeded = True

        if value < lower_threshold and in_event and upper_exceeded:
            in_event = False
            upper_exceeded = False

            lfp_ripple = lfp_det_chans[channel, start_event:idx]
            max_freq = get_event_frequency(lfp_ripple, sampling_rate)

            bandpower_ripple = bandpower(
                lfp_det_chans[channel, start_event:idx],
                sampling_rate,
                RIPPLE_BAND[0],
                RIPPLE_BAND[1],
                "welch",
            )

            # save raw LFP around ripple
            if start_event > 2500 & idx < (lfp_det_chans.shape[1] - 2500):
                raw_lfp = lfp_det_chans[channel, peak_idx - 2500 : peak_idx + 2500]
            else:
                raw_lfp = []

            # only detect resting ripples
           if np.all(resting_ind[start_event:idx]):
                candidate_events.append(
                    CandidateEvent(
                        onset=start_event,
                        offset=idx,
                        peak_amplitude=ripple_band_unsmoothed[peak_idx],
                        peak_idx=peak_idx,
                        detection_channel=detection_channels_ca1[channel],
                        frequency=max_freq,
                        bandpower_ripple=bandpower_ripple,
                        raw_lfp=raw_lfp,
                    )
                )
            peak_amp = (
                -np.inf
            )  # this needs to stay here to be able to identify the correct peak index for the next ripple if the peak_amplitude is the same

    return candidate_events


def get_candidate_ripples(
    lfp_det_chans: np.ndarray,
    detection_channels_ca1: List[int],
    resting_ind: np.ndarray,
    sampling_rate: float,
    detection_method: str,
) -> List[List[CandidateEvent]]:
    """Gets candidate ripples from a common average referenced LFP"""
    channel_idx = list(range(0, len(detection_channels_ca1)))
    return [
        detect_ripple_events(
            channel,
            lfp_det_chans,
            detection_channels_ca1,
            resting_ind,
            sampling_rate,
            detection_method,
        )
        for channel in channel_idx
    ]
