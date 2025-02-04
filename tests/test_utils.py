import numpy as np

from ripples.utils import bandpass_filter, forward_fill, mean_across_same_session, get_event_frequency

def test_get_event_frequency()-> None:
    t = np.arange(0, 1, 1/1000)
    test_data=np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.5 * np.sin(2 * np.pi * 160 * t)
    f=get_event_frequency(test_data, 1000)
    expected_f=50
    assert f == expected_f


def test_butterworth() -> None:
    """Suraya advice to make sure the butterworth filter is going along the correct axis"""
    data = np.random.rand(100, 300)
    first_row = np.expand_dims(data[0, :], axis=1).T

    result_whole_matrix = bandpass_filter(data, 10, 20, 100)
    result_first_row = bandpass_filter(first_row, 10, 20, 100)
    assert result_whole_matrix[0, :].all() == result_first_row.all()


def test_forward_fill() -> None:
    arr = np.array([1, np.nan, np.nan, 4, np.nan, 6, np.nan, np.nan, 9])
    result = forward_fill(arr)

    assert np.array_equal(result, np.array([1, 1, 1, 4, 4, 6, 6, 6, 9]))


def test_mean_across_same_session_no_nesting() -> None:
    class Session:
        def __init__(self, session_id: str, data: float):
            self.id = session_id
            self.data = data

    sessions = [Session("1", 10), Session("1", 20), Session("2", 30), Session("2", 40)]

    result = mean_across_same_session(sessions)  # type: ignore
    expected = [
        15.0,
        35.0,
    ]

    assert result == expected, f"Expected {expected} but got {result}"


def test_mean_across_same_session_another_one() -> None:

    class Session:
        def __init__(self, session_id: str, data: float):
            self.id = session_id
            self.data = data

    sessions = [
        Session("1", 10),
        Session("1", 20),
        Session("2", 30),
        Session("2", 40),
        Session("3", 1000),
    ]

    result = mean_across_same_session(sessions)  # type: ignore
    expected = [15.0, 35.0, 1000]

    assert result == expected, f"Expected {expected} but got {result}"
