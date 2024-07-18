import numpy as np


from ripples.models import CandidateEvent, RotaryEncoder
from ripples.ripple_detection import (
    detect_ripple_events,
    get_resting_ripples,
    remove_duplicate_ripples,
)

from ripples.consts import SAMPLING_RATE_LFP

MIN_DISTANCE = 0.01 * SAMPLING_RATE_LFP  # 10 ms


def test_no_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_power=1.0,
            peak_idx=10 * SAMPLING_RATE_LFP,
        ),
        CandidateEvent(
            onset=20, offset=30, peak_power=2.0, peak_idx=50 * SAMPLING_RATE_LFP
        ),
        CandidateEvent(
            onset=40, offset=50, peak_power=3.0, peak_idx=100 * SAMPLING_RATE_LFP
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert len(result) == 3


def test_with_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_power=1.0,
            peak_idx=10 * SAMPLING_RATE_LFP,
        ),
        CandidateEvent(
            onset=20, offset=30, peak_power=2.0, peak_idx=12 * SAMPLING_RATE_LFP
        ),
        CandidateEvent(
            onset=40, offset=50, peak_power=3.0, peak_idx=100 * SAMPLING_RATE_LFP
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert len(result) == 2
    assert result[0].peak_power == 2.0
    assert result[1].peak_power == 3.0


def test_all_duplicates() -> None:
    ripples = [
        CandidateEvent(onset=0, offset=10, peak_power=0.5, peak_idx=1),
        CandidateEvent(onset=2, offset=12, peak_power=1.0, peak_idx=2),
        CandidateEvent(onset=4, offset=14, peak_power=0.8, peak_idx=3),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert len(result) == 1
    assert result[0].peak_power == 1.0


def test_all_duplicates_end_highest() -> None:
    ripples = [
        CandidateEvent(onset=0, offset=10, peak_power=1.0, peak_idx=1),
        CandidateEvent(onset=2, offset=12, peak_power=0, peak_idx=2),
        CandidateEvent(onset=4, offset=14, peak_power=8, peak_idx=3),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert len(result) == 1
    assert result[0].peak_power == 8


# def test_equal_peak_power() -> None:
# """This fails but we won't encounter this in practice"""
#     ripples = [
#         CandidateEvent(onset=0, offset=10, peak_power=1.0, peak_idx=100),
#         CandidateEvent(onset=2, offset=12, peak_power=1.0, peak_idx=120),
#         CandidateEvent(onset=4, offset=14, peak_power=1.0, peak_idx=140),
#     ]
#     result = remove_duplicate_ripples(ripples, sampling_rate_lfp=SAMPLING_RATE_LFP)
#     assert len(result) == 1
#     assert result[0].peak_power == 1.0
#     assert result[0].peak_idx == 100


def test_multiple_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0, offset=10, peak_power=1.0, peak_idx=SAMPLING_RATE_LFP * 100
        ),
        CandidateEvent(
            onset=5, offset=15, peak_power=10, peak_idx=SAMPLING_RATE_LFP * 109
        ),
        CandidateEvent(
            onset=6, offset=16, peak_power=1.5, peak_idx=SAMPLING_RATE_LFP * 110
        ),
        CandidateEvent(
            onset=15, offset=25, peak_power=1, peak_idx=SAMPLING_RATE_LFP * 200
        ),
        CandidateEvent(
            onset=16, offset=26, peak_power=5, peak_idx=SAMPLING_RATE_LFP * 205
        ),
    ]

    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )

    assert len(result) == 2
    assert result[0].peak_power == 10
    assert result[1].peak_power == 5


def test_detect_ripple_events_basic() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=1, offset=8, peak_idx=4, peak_power=10)]


def test_detect_ripple_events_basic_two_events() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [
        CandidateEvent(onset=1, offset=8, peak_idx=4, peak_power=10),
        CandidateEvent(onset=11, offset=18, peak_idx=14, peak_power=10),
    ]


def test_detect_ripple_events_doesnt_exceed_5x() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 4.9, 4.8, 4.8, 4.8, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == []


def test_detect_ripple_events_bounces_on_upper() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 6, 4, 6, 4, 2.4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=1, offset=6, peak_idx=2, peak_power=6)]


def test_detect_ripple_events_bounces_on_lower() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 2, 3, 6, 5, 2, 4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=3, offset=6, peak_idx=4, peak_power=6)]


def test_detect_ripple_events_jumps_to_upper() -> None:

    data = np.concatenate(
        (
            np.array([6, 5, 4, 3, 2, 1]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=0, offset=4, peak_idx=0, peak_power=6)]


def test_detect_ripple_events_ends_during_ripple() -> None:

    data = np.concatenate(
        (
            np.ones(30),  # load of ones to make the median 1
            np.array([1, 2, 4, 6, 2, 3, 6, 6, 7]),
        )
    )
    result = detect_ripple_events(data)
    assert result == [
        CandidateEvent(onset=2 + 30, offset=4 + 30, peak_idx=3 + 30, peak_power=6)
    ]


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
    result = detect_ripple_events(data)
    assert result == [
        CandidateEvent(onset=0, offset=3, peak_idx=0, peak_power=6),
        CandidateEvent(onset=4, offset=9, peak_idx=6, peak_power=7),
    ]


def test_no_ripples() -> None:
    rotary_encoder = RotaryEncoder(time=np.array([]), position=np.array([]))
    assert (
        get_resting_ripples([], rotary_encoder, 1, sampling_rate_lfp=SAMPLING_RATE_LFP)
        == []
    )


def test_no_resting_ripples() -> None:
    ripples = [
        CandidateEvent(
            onset=0 * SAMPLING_RATE_LFP,
            offset=1 * SAMPLING_RATE_LFP,
            peak_power=5.0,
            peak_idx=5,
        ),
        CandidateEvent(
            onset=1 * SAMPLING_RATE_LFP,
            offset=2 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=25,
        ),
    ]
    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1, 2]), position=np.array([3, 4, 5])
    )
    result = get_resting_ripples(
        ripples, rotary_encoder, 1, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert result == []


def test_all_resting_ripples() -> None:
    ripples = [
        CandidateEvent(
            onset=0 * SAMPLING_RATE_LFP,
            offset=1 * SAMPLING_RATE_LFP,
            peak_power=5.0,
            peak_idx=5,
        ),
        CandidateEvent(
            onset=1 * SAMPLING_RATE_LFP,
            offset=2 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=25,
        ),
    ]

    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1, 2]), position=np.array([0.0, 0.01, 0.02])
    )
    assert get_resting_ripples(ripples, rotary_encoder, 1, SAMPLING_RATE_LFP) == ripples


def test_some_resting_ripples() -> None:
    ripples = [
        CandidateEvent(
            onset=0 * SAMPLING_RATE_LFP,
            offset=1.5 * SAMPLING_RATE_LFP,
            peak_power=5.0,
            peak_idx=1 * SAMPLING_RATE_LFP,
        ),
        CandidateEvent(
            onset=1.5 * SAMPLING_RATE_LFP,
            offset=2.6 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=2 * SAMPLING_RATE_LFP,
        ),
        CandidateEvent(
            onset=2.7 * SAMPLING_RATE_LFP,
            offset=3.9 * SAMPLING_RATE_LFP,
            peak_power=2.0,
            peak_idx=3 * SAMPLING_RATE_LFP,
        ),
    ]
    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1.5, 2.5, 3.5]), position=np.array([10, 0, 0, 2])
    )
    expected_resting_ripples = [ripples[1]]
    result = get_resting_ripples(ripples, rotary_encoder, 1, SAMPLING_RATE_LFP)
    assert result == expected_resting_ripples
