from typing import List
import pytest
import numpy as np

from ripples.consts import SAMPLING_RATE_LFP
from ripples.models import ClusterType
from ripples.multi_session import number_of_spikes_per_cell_per_ripple


# Mock classes to simulate the input data
class Event:
    def __init__(self, peak_idx: int) -> None:
        self.peak_idx = peak_idx


class RipplesSummary:
    def __init__(self, events: List[Event]) -> None:
        self.events = events


class Cluster:
    def __init__(self, info: ClusterType, region: str, spike_times: List[int]) -> None:
        self.info = info
        self.region = region
        self.spike_times = spike_times


class Session:
    def __init__(
        self, ripples_summary: RipplesSummary, clusters_info: List[Cluster]
    ) -> None:
        self.ripples_summary = ripples_summary
        self.clusters_info = clusters_info


def test_number_of_spikes_per_ripple_but_no_valid_clusters() -> None:
    # Simulate ripples but no valid clusters (region not CA1 or not GOOD)
    ripples = [Event(peak_idx=1000), Event(peak_idx=2000)]
    clusters = [
        Cluster(info=ClusterType.NOISE, region="CA1", spike_times=[1500]),
        Cluster(info=ClusterType.GOOD, region="DG", spike_times=[1500]),
    ]
    session = Session(RipplesSummary(ripples), clusters)
    result = number_of_spikes_per_cell_per_ripple(session)  # type: ignore
    assert result == 0.0


def test_number_of_spikes_per_ripple_with_one_valid_cluster() -> None:
    # Simulate ripples and one valid cluster with spikes within ripple windows
    ripples = [
        Event(peak_idx=1000 * SAMPLING_RATE_LFP),
        Event(peak_idx=2000 * SAMPLING_RATE_LFP),
    ]
    spikes = [999.8, 1000.1, 2000.29]  # Some spikes are within 300 ms of ripple peaks
    clusters = [Cluster(info=ClusterType.GOOD, region="CA1", spike_times=spikes)]
    session = Session(RipplesSummary(ripples), clusters)
    result = number_of_spikes_per_cell_per_ripple(session)  # type: ignore
    assert result == 1.5, f"Expected average spikes per ripple to be 1.5, got {result}."


def test_number_of_spikes_per_ripple_with_multiple_valid_clusters() -> None:
    # Simulate ripples and multiple valid clusters
    ripples = [
        Event(peak_idx=1000 * SAMPLING_RATE_LFP),
        Event(peak_idx=2000 * SAMPLING_RATE_LFP),
    ]
    spikes1 = [999.9, 1000.1, 2000.27, 2000.32]  # 3 in ripples
    spikes2 = [999.9, 2000.1, 2000.28, 2000.31]  # 3 in ripples
    spikes3 = [999.9, 2000.1, 2000.28, 2000.27, 1000.1]  # 5 in ripples
    clusters = [
        Cluster(info=ClusterType.GOOD, region="CA1", spike_times=spikes1),
        Cluster(info=ClusterType.GOOD, region="CA1", spike_times=spikes2),
        Cluster(info=ClusterType.GOOD, region="CA1", spike_times=spikes3),
    ]
    session = Session(RipplesSummary(ripples), clusters)
    result = number_of_spikes_per_cell_per_ripple(session)  # type: ignore

    # Three spikes in total during ripple per cluster. Two ripples
    expected = ((3 + 3 + 5) / len(ripples)) / 3
    assert (
        result == expected
    ), f"Expected average spikes per ripple to be {expected}, got {result}."


def test_valid_clusters_but_no_spikes_within_ripples() -> None:
    # Simulate valid clusters but no spikes within ripple windows
    ripples = [Event(peak_idx=1000), Event(peak_idx=2000)]
    spikes = [500, 3000]  # No spikes within 300 ms of ripple peaks
    clusters = [Cluster(info=ClusterType.GOOD, region="CA1", spike_times=spikes)]
    session = Session(RipplesSummary(ripples), clusters)
    result = number_of_spikes_per_cell_per_ripple(session)  # type: ignore
    assert result == 0.0, "Expected 0.0 when no spikes fall within ripple windows."
