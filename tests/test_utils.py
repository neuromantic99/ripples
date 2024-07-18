import numpy as np

from ripples.utils import bandpass_filter, forward_fill


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
