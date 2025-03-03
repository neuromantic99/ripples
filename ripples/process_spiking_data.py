import numpy as np
from typing import Any, Dict, List, Tuple
from scipy.ndimage.filters import gaussian_filter1d

# adapted from ecephys_spike_sorting github repository
# https://github.com/AllenInstitute/ecephys_spike_sorting/tree/main/ecephys_spike_sorting/modules/quality_metrics


def firing_rate(
    spike_train: np.ndarray, min_time: Any = None, max_time: Any = None
) -> float:
    """Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """

    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)

    fr = spike_train.size / duration

    return fr


def isi_violations(
    spike_train: np.ndarray,
    min_time: float,
    max_time: float,
    isi_threshold: float,
    min_isi: float = 0,
) -> Tuple[float, int]:
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations

    """

    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_spikes = len(spike_train)
    num_violations = sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = firing_rate(spike_train, min_time, max_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations


def calculate_isi_violations(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    isi_threshold: float = 0.0015,
    min_isi: float = 0.0005,
) -> np.ndarray:

    cluster_ids = np.unique(spike_clusters)
    total_units = len(cluster_ids)

    viol_rates = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        for_this_cluster = spike_clusters == cluster_id
        viol_rates[idx], num_violations = isi_violations(
            spike_times[for_this_cluster],
            min_time=np.min(spike_times),
            max_time=np.max(spike_times),
            isi_threshold=isi_threshold,
            min_isi=min_isi,
        )

    return viol_rates


def amplitude_cutoff(
    amplitudes: np.ndarray,
    num_histogram_bins: int = 500,
    histogram_smoothing_value: int = 3,
) -> float:
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)

    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible

    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def calculate_amplitude_cutoff(
    spike_clusters: np.ndarray, amplitudes: np.ndarray
) -> np.ndarray:

    cluster_ids = np.unique(spike_clusters)
    total_units = len(cluster_ids)

    amplitude_cutoffs = np.zeros((total_units,))

    for idx, cluster_id in enumerate(cluster_ids):

        for_this_cluster = spike_clusters == cluster_id
        amplitude_cutoffs[idx] = amplitude_cutoff(amplitudes[for_this_cluster])

    return amplitude_cutoffs


def get_good_unit_ids(
    spike_clusters: np.ndarray,
    amplitudes: np.ndarray,
    spike_times: np.ndarray,
    threshold_amplitude_cutoff: float = 0.1,
    threshold_isi_violations: float = 0.5,
) -> np.ndarray:
    amplitude_cutoffs = calculate_amplitude_cutoff(spike_clusters, amplitudes)
    viol_rates = calculate_isi_violations(spike_times, spike_clusters)

    good_unit_ids = (viol_rates < threshold_isi_violations) & (
        amplitude_cutoffs < threshold_amplitude_cutoff
    )

    return good_unit_ids
