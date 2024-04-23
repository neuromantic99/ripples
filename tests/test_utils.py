import numpy as np

from ripples.utils import CandidateEvent, bandpass_filter, detect_ripple_events


def test_detect_ripple_events_basic():

    data = np.concatenate(
        (
            np.array([2, 3, 3, 5, 10, 5, 5, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=1, offset=8, peak_idx=4, peak_power=10)]


def test_detect_ripple_events_basic_two_events():

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


def test_detect_ripple_events_doesnt_exceed_5x():

    data = np.concatenate(
        (
            np.array([2, 3, 3, 4.9, 4.8, 4.8, 4.8, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == []


def test_detect_ripple_events_bounces_on_upper():

    data = np.concatenate(
        (
            np.array([2, 3, 6, 4, 6, 4, 2.4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=1, offset=6, peak_idx=2, peak_power=6)]


def test_detect_ripple_events_bounces_on_lower():

    data = np.concatenate(
        (
            np.array([2, 3, 2, 3, 6, 5, 2, 4, 3, 2, 1.5]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=3, offset=6, peak_idx=4, peak_power=6)]


def test_detect_ripple_events_jumps_to_upper():

    data = np.concatenate(
        (
            np.array([6, 5, 4, 3, 2, 1]),
            np.ones(30),  # load of ones to make the median 1
        )
    )
    result = detect_ripple_events(data)
    assert result == [CandidateEvent(onset=0, offset=4, peak_idx=0, peak_power=6)]


def test_detect_ripple_events_ends_during_ripple():

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


def test_detect_ripple_events_starts_during_ripple():
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


def test_butterworth():
    """Suraya advice to make sure the butterworth filter is going along the correct axis"""
    data = np.random.rand(100, 300)
    first_row = np.expand_dims(data[0, :], axis=1).T

    result_whole_matrix = bandpass_filter(data, 10, 20, 100)
    result_first_row = bandpass_filter(first_row, 10, 20, 100)
    assert result_whole_matrix[0, :].all() == result_first_row.all()
