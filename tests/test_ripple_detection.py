import numpy as np
from numpy import testing
from scipy import io


from ripples.models import CandidateEvent, RotaryEncoder
from ripples.ripple_detection import (
    detect_ripple_events,
    get_resting_ripples,
    remove_duplicate_ripples,
    rotary_encoder_percentage_resting,
    do_preprocessing_lfp_for_ripple_analysis,
)

from unittest.mock import MagicMock, patch

from ripples.consts import SAMPLING_RATE_LFP

MIN_DISTANCE = 0.01 * SAMPLING_RATE_LFP  # 10 ms


def test_no_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_power=1.0,
            peak_idx=10 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=20,
            offset=30,
            peak_power=2.0,
            peak_idx=50 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=40,
            offset=50,
            peak_power=3.0,
            peak_idx=100 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
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
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=20,
            offset=30,
            peak_power=2.0,
            peak_idx=12 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=40,
            offset=50,
            peak_power=3.0,
            peak_idx=100 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
    ]
    result = remove_duplicate_ripples(
        ripples,
        min_distance_seconds=MIN_DISTANCE,
        sampling_rate_lfp=SAMPLING_RATE_LFP,
    )
    assert len(result) == 2
    assert result[0].peak_power == 2.0
    assert result[1].peak_power == 3.0


def test_all_duplicates() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_power=0.5,
            peak_idx=1,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=2,
            offset=12,
            peak_power=1.0,
            peak_idx=2,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=4,
            offset=14,
            peak_power=0.8,
            peak_idx=3,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
    ]
    result = remove_duplicate_ripples(
        ripples, min_distance_seconds=MIN_DISTANCE, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert len(result) == 1
    assert result[0].peak_power == 1.0


def test_all_duplicates_end_highest() -> None:
    ripples = [
        CandidateEvent(
            onset=0,
            offset=10,
            peak_power=1.0,
            peak_idx=1,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=2,
            offset=12,
            peak_power=0,
            peak_idx=2,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=4,
            offset=14,
            peak_power=8,
            peak_idx=3,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
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
            onset=0,
            offset=10,
            peak_power=1.0,
            peak_idx=SAMPLING_RATE_LFP * 100,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=5,
            offset=15,
            peak_power=10,
            peak_idx=SAMPLING_RATE_LFP * 109,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=6,
            offset=16,
            peak_power=1.5,
            peak_idx=SAMPLING_RATE_LFP * 110,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=15,
            offset=25,
            peak_power=1,
            peak_idx=SAMPLING_RATE_LFP * 200,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=16,
            offset=26,
            peak_power=5,
            peak_idx=SAMPLING_RATE_LFP * 205,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
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
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 1
        assert result[0].offset == 8
        assert result[0].peak_idx == 4
        assert result[0].peak_power == 10


def test_detect_ripple_events_basic_two_events() -> None:

    data = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    CA1_channels = [200, 201]

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 1
        assert result[0].offset == 8
        assert result[0].peak_idx == 4
        assert result[0].peak_power == 10

        assert result[1].onset == 11
        assert result[1].offset == 18
        assert result[1].peak_idx == 14
        assert result[1].peak_power == 10


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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)
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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 1
        assert result[0].offset == 6
        assert result[0].peak_idx == 2
        assert result[0].peak_power == 6


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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 3
        assert result[0].offset == 6
        assert result[0].peak_idx == 4
        assert result[0].peak_power == 6


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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 0
        assert result[0].offset == 4
        assert result[0].peak_idx == 0
        assert result[0].peak_power == 6


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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 2 + 30
        assert result[0].offset == 4 + 30
        assert result[0].peak_idx == 3 + 30
        assert result[0].peak_power == 6


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

    with patch(
        "ripples.ripple_detection.do_preprocessing_lfp_for_ripple_analysis",
        lambda data, y, z: data[0, :],
    ):
        result = detect_ripple_events(0, data, CA1_channels, 2500)

        assert result[0].onset == 0
        assert result[0].offset == 3
        assert result[0].peak_idx == 0
        assert result[0].peak_power == 6

        assert result[1].onset == 4
        assert result[1].offset == 9
        assert result[1].peak_idx == 6
        assert result[1].peak_power == 7


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
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=1 * SAMPLING_RATE_LFP,
            offset=2 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=25,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
    ]
    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1, 2, 3]), position=np.array([3, 4, 5, 6])
    )
    result = get_resting_ripples(
        ripples, rotary_encoder, 1, sampling_rate_lfp=SAMPLING_RATE_LFP
    )
    assert result == []


def test_all_resting_ripples() -> None:
    ripples = [
        CandidateEvent(
            onset=1 * SAMPLING_RATE_LFP,
            offset=2 * SAMPLING_RATE_LFP,
            peak_power=5.0,
            peak_idx=5,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        CandidateEvent(
            onset=2 * SAMPLING_RATE_LFP,
            offset=3 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=25,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
    ]

    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1, 2, 3, 4]), position=np.array([0.0, 0.01, 0.02, 0.03, 0.04])
    )
    assert get_resting_ripples(ripples, rotary_encoder, 1, SAMPLING_RATE_LFP) == ripples


def test_some_resting_ripples() -> None:
    ripples = [
        # to exclude because at the beginning of the recording, not allowing calculation of speed
        CandidateEvent(
            onset=0 * SAMPLING_RATE_LFP,
            offset=1.5 * SAMPLING_RATE_LFP,
            peak_power=5.0,
            peak_idx=1 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        # resting state
        CandidateEvent(
            onset=1.5 * SAMPLING_RATE_LFP,
            offset=2.6 * SAMPLING_RATE_LFP,
            peak_power=8.0,
            peak_idx=2 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
        # locomotion
        CandidateEvent(
            onset=2.7 * SAMPLING_RATE_LFP,
            offset=3.9 * SAMPLING_RATE_LFP,
            peak_power=2.0,
            peak_idx=3 * SAMPLING_RATE_LFP,
            frequency=1,
            bandpower_ripple=1,
            detection_channel=1,
        ),
    ]
    rotary_encoder = RotaryEncoder(
        time=np.array([0, 1.5, 2.5, 3.5, 4.5]), position=np.array([10, 10, 10, 10, 40])
    )
    expected_resting_ripples = [ripples[1]]
    result = get_resting_ripples(ripples, rotary_encoder, 1, SAMPLING_RATE_LFP)
    assert result == expected_resting_ripples


# Test 1: Test with normal speed data and no rest periods
def test_rotary_encoder_percentage_resting_above_threshold() -> None:
    # Mock RotaryEncoder with increasing position over time
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 1, 2, 3, 4, 5])

    # Threshold below the minimum speed, max_time larger than the time array
    result = rotary_encoder_percentage_resting(
        rotary_encoder, threshold=0.9, max_time=6
    )
    assert result[0] == 0.0, "Expected no resting periods but got some."


def test_rotary_encoder_percentage_resting_all_rest() -> None:
    # Mock RotaryEncoder with no movement (stationary)
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 0, 0, 0, 0, 0])

    # Threshold above the maximum speed (speed is 0 everywhere)
    result = rotary_encoder_percentage_resting(
        rotary_encoder, threshold=0.1, max_time=6
    )
    assert result[0] == 1, "Expected all resting period"


# Test 3: Test with mixed movement (part resting, part moving)
def test_rotary_encoder_percentage_resting_mixed() -> None:
    # Mock RotaryEncoder with some movement and some stationary
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 0, 0, 1, 2, 4])

    result = rotary_encoder_percentage_resting(
        rotary_encoder, threshold=1.1, max_time=6
    )
    assert result[0] == 0.8


def test_rotary_encoder_percentage_resting_at_end() -> None:
    # Mock RotaryEncoder with stationary data
    rotary_encoder = MagicMock()
    rotary_encoder.time = np.array([0, 1, 2, 3, 4, 5])
    rotary_encoder.position = np.array([0, 10, 20, 30, 40, 50])

    result = rotary_encoder_percentage_resting(
        rotary_encoder, threshold=0.1, max_time=10
    )

    # Does not include the max time itself which maybe is not the correct behaviour but
    # won't make a difference
    assert result[0] == 4 / 9


def test_do_preprocessing_lfp_for_ripple_analysis() -> None:

    t = np.arange(0, 1, 1 / 2500)
    ripple = (
        np.sin(2 * np.pi * 130 * t)
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

    sm_envelope = do_preprocessing_lfp_for_ripple_analysis(data, 2500, 0)
    m = io.loadmat("C:/Python_code/ripples/matlab_comparison_ripple_detection.mat")

    testing.assert_allclose(
        m["data_m"][:, 1000:21500], data[1, 1000:21500].reshape(1, 20500)
    )
    testing.assert_allclose(
        m["sm_envelope_m"][:, 1000:21500],
        sm_envelope[1000:21500].reshape(1, 20500),
        atol=0.012,
    )
    # Set the absolute tolerance of difference to the matlab results to 0.012, should be just passing this test


def test_do_preprocessing_lfp_for_ripple_analysis_real_ripple() -> None:

    m = io.loadmat(
        "C:/Python_code/ripples/matlab_comparison_ripple_detection_real_ripple.mat"
    )

    data = m["data_m"]

    data = np.vstack(
        [data, data]
    )  # need to mimic an array with at leat two channels for the code to work

    sm_envelope = do_preprocessing_lfp_for_ripple_analysis(data, 2500, 0)

    testing.assert_allclose(
        m["sm_envelope_m"][:, 1000:5500],
        sm_envelope.reshape(1, 6500)[:, 1000:5500],
        rtol=2.6,
        atol=5,
    )
    # Set the absolute and relative tolerance of difference so that it should be just passing this test
