import numpy as np


def split_vector_consecutive(arr):
    """
    Splits the input array into sub-arrays of consecutive integers.

    Parameters:
    - arr: A numpy array of integers.

    Returns:
    - A list of numpy arrays, each containing a sequence of consecutive integers from the input array.

    TODO: test this
    """

    # Create an array of consecutive integers starting from 1
    n = np.arange(1, len(arr) + 1)
    # Calculate the difference to find breaks in consecutive sequences
    diff = np.diff(arr - n)
    # Identify where the differences change, indicating the end of a consecutive sequence
    split_indices = np.where(diff != 0)[0] + 1
    # Split the array at the identified indices
    return np.split(arr, split_indices)
