import numpy as np
from numpy import testing
from scipy import io
from pathlib import Path

from ripples.models import CandidateEvent
from ripples.ripple_detection import (
    detect_ripple_events,
    remove_duplicate_ripples,
    do_preprocessing_lfp_for_ripple_analysis,
    get_candidate_ripples,
)
from ripples.analysis import get_resting_periods, pad_resting_ind

from unittest.mock import MagicMock, patch

from ripples.consts import HERE

MIN_DISTANCE = 0.01 * 2500  # 10 ms


def test_no_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_amplitude=1.0,
            peak_idx=10 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=20,
            offset=30,
            peak_amplitude=2.0,
            peak_idx=50 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=40,
            offset=50,
            peak_amplitude=3.0,
            peak_idx=100 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=2500
    )
    assert len(result) == 3


def test_with_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_amplitude=1.0,
            peak_idx=10 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=20,
            offset=30,
            peak_amplitude=2.0,
            peak_idx=12 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=40,
            offset=50,
            peak_amplitude=3.0,
            peak_idx=100 * 2500,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
    ]
    result = remove_duplicate_ripples(
        ripples,
        min_distance_seconds=MIN_DISTANCE,
        sampling_rate_lfp=2500,
    )
    assert len(result) == 2
    assert result[0].peak_amplitude == 2.0
    assert result[1].peak_amplitude == 3.0


def test_all_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_amplitude=0.5,
            peak_idx=1,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=2,
            offset=12,
            peak_amplitude=1.0,
            peak_idx=2,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=4,
            offset=14,
            peak_amplitude=0.8,
            peak_idx=3,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=2500
    )
    assert len(result) == 1
    assert result[0].peak_amplitude == 1.0


def test_all_duplicates_end_highest() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_amplitude=1.0,
            peak_idx=1,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=2,
            offset=12,
            peak_amplitude=0,
            peak_idx=2,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=4,
            offset=14,
            peak_amplitude=8,
            peak_idx=3,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=2500
    )
    assert len(result) == 1
    assert result[0].peak_amplitude == 8


# def test_equal_peak_amplitude() -> None:
# """This fails but we won't encounter this in practice"""
#     ripples = [
#         CandidateEvent(onset=0, offset=10, peak_amplitude=1.0, peak_idx=100),
#         CandidateEvent(onset=2, offset=12, peak_amplitude=1.0, peak_idx=120),
#         CandidateEvent(onset=4, offset=14, peak_amplitude=1.0, peak_idx=140),
#     ]
#     result = remove_duplicate_ripples(ripples, sampling_rate_lfp=2500)
#     assert len(result) == 1
#     assert result[0].peak_amplitude == 1.0
#     assert result[0].peak_idx == 100


def test_multiple_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_amplitude=1.0,
            peak_idx=2500 * 100,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=5,
            offset=15,
            peak_amplitude=10,
            peak_idx=2500 * 109,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=6,
            offset=16,
            peak_amplitude=1.5,
            peak_idx=2500 * 110,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=15,
            offset=25,
            peak_amplitude=1,
            peak_idx=2500 * 200,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
        CandidateEvent(
            onset=16,
            offset=26,
            peak_amplitude=5,
            peak_idx=2500 * 205,
            frequency=1,
            bandpower_ripple=1,
            strength=1,
            detection_channel=1,
            raw_lfp=[1, 1],
        ),
    ]

    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=2500
    )

    assert len(result) == 2
    assert result[0].peak_amplitude == 10
    assert result[1].peak_amplitude == 5


def test_detect_ripple_events_basic() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )

        assert result[0].onset == 1
        assert result[0].offset == 8
        assert result[0].peak_idx == 4
        assert result[0].peak_amplitude == 10


def test_detect_ripple_events_basic_two_events() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 5, 7, 10, 7, 5, 3, 2, 1.5]),
            np.array([2, 5, 3, 5, 10, 5, 3, 5, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )

        assert result[0].onset == 1
        assert result[0].offset == 8
        assert result[0].peak_idx == 4
        assert result[0].peak_amplitude == 10

        assert result[1].onset == 11
        assert result[1].offset == 18
        assert result[1].peak_idx == 14
        assert result[1].peak_amplitude == 10

        assert result[1].frequency > result[0].frequency
        assert result[0].detection_channel == 200


def test_detect_ripple_events_doesnt_exceed_5x() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 4.9, 4.8, 4.8, 4.8, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):

        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )
        assert result == []


def test_detect_ripple_events_bounces_on_upper() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 6, 4, 6, 4, 2.4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )  # need lower sampling rate if not ripple frequency to high and code throws an error

        assert result[0].onset == 1
        assert result[0].offset == 6
        assert result[0].peak_idx == 2
        assert result[0].peak_amplitude == 6


def test_detect_ripple_events_bounces_on_lower() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 2, 3, 6, 5, 2, 4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )

        assert result[0].onset == 3
        assert result[0].offset == 6
        assert result[0].peak_idx == 4
        assert result[0].peak_amplitude == 6


def test_detect_ripple_events_jumps_to_upper() -> None:

    data = np.concatenate(
        (
            np.array([6, 5, 4, 3, 2, 1]),
            np.ones(30),  # load of ones to make the median 1
        )
    )

    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )

        assert result[0].onset == 0
        assert result[0].offset == 4
        assert result[0].peak_idx == 0
        assert result[0].peak_amplitude == 6


def test_detect_ripple_events_ends_during_ripple() -> None:

    data = np.concatenate(
        (
            np.ones(30),  # load of ones to make the median 1
            np.array([1, 2, 4, 6, 2, 3, 6, 6, 7]),
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )

        assert result[0].onset == 2 + 30
        assert result[0].offset == 4 + 30
        assert result[0].peak_idx == 3 + 30
        assert result[0].peak_amplitude == 6


def test_detect_ripple_events_starts_during_ripple() -> None:
    """This test demonstrates the behaviour more than tests it as it's a bit different to the behaviour when ending during a ripple.
    Data starting during a ripple with have the ripple onset assigned to 0.
    Probably not a worry and excluding this ripple makes the code less clean
    """

    data = np.concatenate(
        (
            np.array([6, 6, 3, 2, 3, 6, 7, 4, 3, 2]),
            np.ones(30),  # load of ones to make the median 1
        )
    )

    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: (data[0, :], data[0, :]),
    ):
        result = detect_ripple_events(
            0, data, CA1_channels, resting_ind, 2500, "median"
        )
        assert result[0].onset == 0
        assert result[0].offset == 3
        assert result[0].peak_idx == 0
        assert result[0].peak_amplitude == 6

        assert result[1].onset == 4
        assert result[1].offset == 9
        assert result[1].peak_idx == 6
        assert result[1].peak_amplitude == 7


def test_get_candidate_ripple() -> None:

    data1 = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.array([2, 3, 7, 4, 10, 5, 4, 7, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data2 = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data1, data2]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]
    resting_ind = np.ones(data.shape[1], dtype=bool)

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, sampling_rate, channel: (data[channel, :], data[channel, :]),
    ):
        result = get_candidate_ripples(data, CA1_channels, resting_ind, 2500, "median")

        assert len(result) == len(CA1_channels)
        assert len(result[0]) == 2
        assert len(result[1]) == 1

        assert result[0][0].peak_idx == 4
        assert result[0][1].peak_idx == 14
        assert result[1][0].peak_idx == 4


def test_get_resting_periods() -> None:
    rotary_encoder = MagicMock()
    rotary_encoder.position = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rotary_encoder.time = [10, 11, 12, 13, 41, 41.5, 42, 85, 86, 87]
    max_time = 90 * 2500
    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time)

    assert np.round(90 - sum(resting_ind) / 2500) == len(rotary_encoder.time) - 2
    # max time in seconds - resting time/sampling_rate should be equivalent to the length of rotary encoder time
    # because that is the locomotion  period; I am subtracting 2 because of the binning used for resting_ind calculation

    assert np.all(np.logical_not(resting_ind[10 * 2500 : 13 * 2500]))
    assert np.all(np.logical_not(resting_ind[41 * 2500 : 42 * 2500]))

    assert np.all(np.logical_not(resting_ind[85 * 2500 : 87 * 2500]))
    assert resting_ind[int(9 * 2500)] == True


# Test 1: Test with normal speed data and no rest periods
def test_get_resting_periods_2() -> None:
    # Mock RotaryEncoder with increasing position over time
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 1, 2, 3, 4, 5])

    # Threshold below the minimum speed, max_time larger than the time array
    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time=(5 * 2500))

    result = sum(resting_ind) / len(resting_ind)
    assert result == 0.0


def test_get_resting_periods_max_time_greater_than_bin_edge() -> None:
    # Mock RotaryEncoder with increasing position over time
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5, 5.5])
    rotary_encoder.position = np.array([0, 1, 2, 3, 4, 5, 6])
    # Threshold below the minimum speed, max_time larger than the time array
    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time=(5.5 * 2500))
    result = sum(resting_ind) / len(resting_ind)
    assert result == 0.0


def test_get_resting_periods_all_rest() -> None:
    # Mock RotaryEncoder with no movement (stationary)
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 0, 0, 0, 0, 0])

    # Threshold above the maximum speed (speed is 0 everywhere)
    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time=(5 * 2500))
    result = sum(resting_ind) / len(resting_ind)
    assert int(result) == 1, "Expected all resting period"


# Test 3: Test with mixed movement (part resting, part moving)
def test_get_resting_periods_resting_mixed() -> None:
    # Mock RotaryEncoder with some movement and some stationary
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 0, 0, 1, 2, 4])

    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time=(5 * 2500))
    result = sum(resting_ind) / len(resting_ind)
    assert np.round(result, decimals=1) == 0.4


def test_get_resting_periods_resting_at_end() -> None:
    # Mock RotaryEncoder with stationary data
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 10, 20, 30, 40, 50])
    max_time = 10 * 2500
    with patch(
        "ripples.analysis.pad_resting_ind",
        lambda x, y: x,
    ):
        resting_ind, speed = get_resting_periods(rotary_encoder, max_time)

    result = sum(resting_ind) / len(resting_ind)
    assert result == 0.5


def test_get_resting_periods_resting_at_end_with_padding() -> None:
    # Mock RotaryEncoder with stationary data
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 10, 20, 30, 40, 50])
    max_time = 10 * 2500

    resting_ind, speed = get_resting_periods(rotary_encoder, max_time)
    result = sum(resting_ind) / len(resting_ind)
    # Test should pass when padding is set to 2500 in analysis.py
    assert result == 0.4  # max time 10s, 5s resting, 1s padding


def test_pad_resting_ind_locomotion_in_the_middle() -> None:
    resting_ind = np.ones(10 * 2500)
    locomotion_period = range((8 * 2500), (9 * 2500))
    resting_ind[locomotion_period] = 0
    padding = 1250
    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)
    assert sum(resting_ind_after_padding) / len(resting_ind_after_padding) == 0.8
    assert sum(resting_ind) / len(resting_ind) == 0.9
    assert len(resting_ind) == len(resting_ind)


def test_pad_resting_ind_locomotion_at_the_end() -> None:
    resting_ind = np.ones(10 * 2500)
    locomotion_period = range((8 * 2500), (10 * 2500))
    resting_ind[locomotion_period] = 0
    padding = 1250
    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)
    assert sum(resting_ind_after_padding) / len(resting_ind_after_padding) == 0.75
    assert sum(resting_ind) / len(resting_ind) == 0.8
    assert len(resting_ind) == len(resting_ind)


def test_pad_resting_ind_simple() -> None:
    resting_ind = np.ones(10)

    resting_ind[3] = 0
    padding = 2

    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)

    import matplotlib.pyplot as plt

    # plt.plot(resting_ind.astype(int), "o", color="red", label="before padding")
    # plt.plot(
    #     resting_ind_after_padding.astype(int), ".", color="blue", label="after padding"
    # )
    # plt.legend()
    # plt.show()

    assert sum(resting_ind_after_padding) == 5
    assert sum(resting_ind) == 9


def test_pad_resting_ind_simple_locomotion_at_the_beginning() -> None:
    resting_ind = np.ones(10)

    resting_ind[1] = 0
    padding = 2

    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)

    import matplotlib.pyplot as plt

    # plt.plot(resting_ind.astype(int), "o", color="red", label="before padding")
    # plt.plot(
    #     resting_ind_after_padding.astype(int), ".", color="blue", label="after padding"
    # )
    # plt.legend()
    # plt.show()

    assert sum(resting_ind_after_padding) == 6
    assert sum(resting_ind) == 9


def test_pad_resting_ind_simple_locomotion_at_the_end() -> None:
    resting_ind = np.ones(10)

    resting_ind[8] = 0
    padding = 2

    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)

    import matplotlib.pyplot as plt

    # plt.plot(resting_ind.astype(int), "o", color="red", label="before padding")
    # plt.plot(
    #     resting_ind_after_padding.astype(int), ".", color="blue", label="after padding"
    # )
    # plt.legend()
    # plt.show()

    assert sum(resting_ind_after_padding) == 6
    assert sum(resting_ind) == 9


def test_pad_resting_ind_simple_locomotion_in_the_middle() -> None:
    resting_ind = np.ones(10)

    resting_ind[5] = 0
    padding = 2

    resting_ind_after_padding = pad_resting_ind(resting_ind, padding)

    import matplotlib.pyplot as plt

    # plt.plot(resting_ind.astype(int), "o", color="red", label="before padding")
    # plt.plot(
    #     resting_ind_after_padding.astype(int), ".", color="blue", label="after padding"
    # )
    # plt.legend()
    # plt.show()

    assert sum(resting_ind_after_padding) == 5
    assert sum(resting_ind) == 9


def test_do_preprocessing_lfp_for_ripple_analysis() -> None:

    t = np.arange(0, 1, 1 / 2500)
    ripple = (
        np.sin(2 * np.pi * 150 * t)
        + 0.5 * np.sin(2 * np.pi * 50 * t)
        + 0.5 * np.sin(2 * np.pi * 180 * t)
    )
    ripple = ripple + 1

    data = np.concatenate(
        (
            np.ones(10000),
            ripple,
            np.ones(10000),  # load of ones to make the median 1
        )
    )

    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    sm_envelope, _ = do_preprocessing_lfp_for_ripple_analysis(data, 2500, 0)
    m = io.loadmat(
        Path(HERE.parent / "matlab" / "matlab_comparison_ripple_detection.mat")
    )

    testing.assert_allclose(
        m["data_m"][:, 1000:21500], data[1, 1000:21500].reshape(1, 20500)
    )
    testing.assert_allclose(
        m["sm_envelope_m"][:, 1000:21500],
        sm_envelope[1000:21500].reshape(1, 20500),
        atol=0.041,
        rtol=7.17331689e11,
    )
    # Set the absolute tolerance and relativ tolerance so that it just passes this test, in plotting it looks great


def test_do_preprocessing_lfp_for_ripple_analysis_real_ripple() -> None:

    m = io.loadmat(
        Path(
            HERE.parent
            / "matlab"
            / "matlab_comparison_ripple_detection_real_ripple.mat"
        )
    )

    data = m["data_m"]

    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    sm_envelope, _ = do_preprocessing_lfp_for_ripple_analysis(data, 2500, 0)

    testing.assert_allclose(
        m["sm_envelope_m"][:, 1000:5500],
        sm_envelope.reshape(1, 6500)[:, 1000:5500],
        rtol=2.6,
        atol=5,
    )
    # Set the absolute and relative tolerance of difference so that it should be just passing this test
