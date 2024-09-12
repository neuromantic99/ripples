import math
import matplotlib.pyplot as plt
import numpy as np
from ripples.models import ClusterType, Session
from ripples.multi_session import load_sessions

import mrestimator as mre


def bin_spikes(session: Session) -> np.ndarray:

    bin_size = 0.01  # 10ms
    end_time = math.ceil(session.length_seconds)
    bin_edges = np.arange(0, end_time + bin_size, bin_size)

    binned = []
    for cluster in session.clusters_info:

        if cluster.info != ClusterType.GOOD or cluster.region != "CA1":
            continue
        # Create bin edges (start_time to end_time in 10ms intervals)
        # Bin the spike times using np.histogram
        spike_counts, _ = np.histogram(cluster.spike_times, bins=bin_edges)
        binned.append(spike_counts)

    return np.array(binned)


def compute_m(session: Session) -> float:
    binned = bin_spikes(session)

    rk = mre.coefficients(np.mean(binned, axis=0))
    m1 = mre.fit(rk, fitfunc=mre.f_exponential)
    m2 = mre.fit(rk, fitfunc=mre.f_exponential_offset)
    m3 = mre.fit(rk, fitfunc=mre.f_complex)

    # plot manually without using OutputHandler
    plt.plot(rk.steps, rk.coefficients, label="data")
    plt.plot(
        rk.steps,
        mre.f_exponential(rk.steps, *m1.popt),
        label="exponential m={:.5f}".format(m1.mre),
    )
    plt.plot(
        rk.steps,
        mre.f_exponential_offset(rk.steps, *m2.popt),
        label="exp + offset m={:.5f}".format(m2.mre),
    )
    plt.plot(
        rk.steps,
        mre.f_complex(rk.steps, *m3.popt),
        label="complex m={:.5f}".format(m3.mre),
    )

    plt.xlabel("Time (ms)")
    plt.legend()
    plt.show()

    return 0.0


def main() -> None:
    WTs, NLGFs = load_sessions()
    compute_m(NLGFs[0])
