import numpy as np

from ripples.utils import CandidateEvent, bandpass_filter, detect_ripple_events


def test_butterworth():
    """Suraya advice to make sure the butterworth filter is going along the correct axis"""
    data = np.random.rand(100, 300)
    first_row = np.expand_dims(data[0, :], axis=1).T

    result_whole_matrix = bandpass_filter(data, 10, 20, 100)
    result_first_row = bandpass_filter(first_row, 10, 20, 100)
    assert result_whole_matrix[0, :].all() == result_first_row.all()
