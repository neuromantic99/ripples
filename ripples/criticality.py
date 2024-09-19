import math
from typing import List
import matplotlib.pyplot as plt
from pydantic import BaseModel
import numpy as np
import os

os.environ["DISPLAY"] = "s"  # Turn off mrestimator plot window override

from ripples.consts import HERE
from ripples.models import ClusterInfo, ClusterType, Session


import mrestimator as mre
import mat73


class RobSession(BaseModel):
    length_seconds: float
    clusters_info: List[ClusterInfo]


def bin_spikes(session: Session | RobSession) -> np.ndarray:

    bin_size = 0.01  # 10ms
    end_time = math.ceil(session.length_seconds)
    bin_edges = np.arange(0, end_time + bin_size, bin_size)

    binned = []
    for cluster in session.clusters_info:

        if cluster.info != ClusterType.GOOD or "CA1" not in cluster.region:
            # if cluster.info != ClusterType.GOOD or "RSP" not in cluster.region:
            continue
        spike_counts, _ = np.histogram(cluster.spike_times, bins=bin_edges)
        binned.append(spike_counts)

    return np.array(binned)


def compute_m(session: Session | RobSession, plot: bool = False) -> float:
    binned = bin_spikes(session)

    rk = mre.coefficients(np.mean(binned, axis=0))
    m2 = mre.fit(rk, fitfunc=mre.f_exponential_offset)

    if plot:
        m1 = mre.fit(rk, fitfunc=mre.f_exponential)
        m3 = mre.fit(rk, fitfunc=mre.f_complex)
        plt.plot(rk.steps, rk.coefficients, label="data")
        plt.plot(
            rk.steps,
            mre.f_exponential(rk.steps, *m1.popt),
            label=f"exponential m={round(m1.mre, 3)} tau = {round(m1.tau, 3)}",
        )
        plt.plot(
            rk.steps,
            mre.f_exponential_offset(rk.steps, *m2.popt),
            label=f"exp + offset m={round(m2.mre, 3)} tau = {round(m2.tau, 3)}",
        )
        plt.plot(
            rk.steps,
            mre.f_complex(rk.steps, *m3.popt),
            label=f"complex m={round(m3.mre, 3)} tau = {round(m3.tau, 3)}",
        )

        plt.xlabel("Time (ms)")
        plt.legend()
        plt.show()

    return m2.mre


def load_robs_data() -> RobSession:
    probe_details = mat73.loadmat(
        HERE.parent / "data" / "Rob" / "probe_trajectory_details_v2__imec1.mat"
    )["probe_details"]

    area_channel = probe_details["alignedregions"]
    area_channel = np.array(
        [
            channel[0] if channel[0] is not None else "outside_brain"
            for channel in area_channel
        ]
    )
    area_channel = np.repeat(area_channel, 2)
    clusters = mat73.loadmat(
        HERE.parent / "data" / "Rob" / "spike_sorted_clusters_imec1.mat"
    )
    channel_cluster = clusters["out"]["GIU_channel"]
    cluster_area = area_channel[channel_cluster.astype(int)]

    clusters_info = [
        ClusterInfo(
            spike_times=list(clusters["out"]["spikeTimesGIU"][i]),
            region=cluster_area[i],
            info=ClusterType.GOOD,
            depth=0,
            channel=channel_cluster[i],
        )
        for i in range(len(clusters["out"]["spikeTimesGIU"]))
    ]

    return RobSession(
        length_seconds=clusters["out"]["total_recording_time"][0],
        clusters_info=clusters_info,
    )


def main() -> None:
    session = load_robs_data()
    compute_m(session, plot=True)
    # WTs, NLGFs = load_sessions()
    # compute_m(WTs[0], plot=True)
    # wt_data = [compute_m(wt) for wt in WTs]
    # nlgf_data = [compute_m(nlgf) for nlgf in NLGFs]

    # sns.stripplot({"WTs": wt_data, "NLGFs": nlgf_data}, color="black")
    # sns.boxplot({"WTs": wt_data, "NLGFs": nlgf_data}, showfliers=False)
    # plt.ylabel("Spikes per ripple per CA1 neuron")
    # plt.tight_layout()
    # # plt.ylim(0.98, 1.005)
    # plt.axhline(1, color="grey", ls="--", alpha=0.7)
    # plt.text(1, 1, "Criticality", color="grey", alpha=0.7, fontsize=12)
    # # plt.savefig(HERE.parent / "figures" / "spkes_per_ripple.png")
    # plt.show()
    # compute_m(WTs[0])
