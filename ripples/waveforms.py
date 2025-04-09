import numpy as np
import os
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def get_waveforms(gwfparams: dict) -> dict:
    """
    Pulls out the waveforms from the temp_wh.dat file (written by kilsort) and gives the mean waveform for each cluster
    adapted from Matlab Version written by the cortex lab (https://github.com/cortex-lab/spikes/tree/master/analysis)
    """
    file_path = os.path.join(gwfparams["dataDir"], gwfparams["fileName"])
    data_type = np.dtype(gwfparams["dataType"])
    n_ch = gwfparams["nCh"]
    wf_win = np.arange(gwfparams["wfWin"][0], gwfparams["wfWin"][1])
    wf_n_samples = len(wf_win)

    file_size = os.path.getsize(file_path)
    data_type_nbytes = data_type.itemsize
    n_samp = file_size // (n_ch * data_type_nbytes)

    mmf = np.memmap(file_path, dtype=data_type, mode="r", shape=(n_samp, n_ch))

    print("loaded data")

    ch_map = np.load(os.path.join(gwfparams["dataDir"], "channel_map.npy"))[0]
    ch_map = ch_map.astype(int)

    unit_ids = np.unique(gwfparams["spikeClusters"])
    num_units = len(unit_ids)
    spike_time_keeps = np.full((num_units, gwfparams["nWf"]), np.nan)
    wave_forms = np.full(
        (num_units, gwfparams["nWf"], len(ch_map), wf_n_samples), np.nan
    )
    wave_forms_mean_per_channel = np.full(
        (num_units, len(ch_map), wf_n_samples), np.nan
    )
    max_channel_idx = np.full(num_units, np.nan)
    wave_forms_mean_max_channel = np.full((num_units, wf_n_samples), np.nan)
    wave_forms_mean_norm = np.full((num_units, wf_n_samples), np.nan)
    halfwidth_at_third_max = np.full(num_units, np.nan)
    trough_to_peak_time = np.full(num_units, np.nan)
    aligned_wf = np.full((num_units, 150), np.nan)

    for i, unit_id in enumerate(unit_ids):
        cur_spike_times = gwfparams["spikeTimes"][gwfparams["spikeClusters"] == unit_id]
        cur_unit_n_spikes = len(cur_spike_times)

        if cur_unit_n_spikes == 0:
            continue

        spike_times_rp = np.random.permutation(cur_spike_times)
        keep_count = min(gwfparams["nWf"], cur_unit_n_spikes)
        spike_time_keeps[i, :keep_count] = np.sort(
            spike_times_rp[:keep_count]
        ).squeeze()

        for j in range(keep_count):
            spike_time = int(spike_time_keeps[i, j])
            if (
                spike_time + gwfparams["wfWin"][1] >= n_samp
                or spike_time + gwfparams["wfWin"][0] < 0
            ):
                continue

            wave_forms[i, j, :, :] = mmf[spike_time + wf_win][:, ch_map].T.squeeze()

        wave_forms_mean_per_channel[i, :, :] = np.nanmean(
            wave_forms[i, :, :, :], axis=0
        )

        max_channel_idx[i] = np.argmax(
            np.max(np.abs(wave_forms_mean_per_channel[i, :, :]), axis=1)
        )
        wave_forms_mean_max_channel[i, :] = wave_forms_mean_per_channel[
            i, int(max_channel_idx[i]), :
        ]
        wave_forms_mean_norm[i, :] = -1 * (
            wave_forms_mean_max_channel[i, :] / min(wave_forms_mean_max_channel[i, :])
        )

        halfwidth_at_third_max[i] = calculate_waveform_halfwidth(
            wave_forms_mean_norm[i, :], 0.3
        )
        trough_to_peak_time[i] = calculate_waveform_trough_to_peak_time(
            wave_forms_mean_norm[i, :]
        )

        # find trough
        tim_min = np.argmin(wave_forms_mean_norm[i, :])
        # set trough to 75 and fill up with nans
        aligned_wf[i, :] = np.concatenate(
            (
                np.full((75 - tim_min,), wave_forms_mean_norm[i, 0]),
                wave_forms_mean_norm[i, :],
                np.full(
                    (150 - (75 - tim_min) - len(wave_forms_mean_norm[i, :])),
                    wave_forms_mean_norm[i, -1],
                ),
            )
        )

        print(f"Completed {i+1} of {num_units} units.")

    return {
        "unitIDs": unit_ids,
        "waveForms": wave_forms,
        "waveFormsMeanChannel": wave_forms_mean_per_channel,
        "waveFormsMean_max_channel": wave_forms_mean_max_channel,
        "max_channel": np.array(max_channel_idx, dtype=int),
        "waveFormsMean_norm": wave_forms_mean_norm,
        "aligned_wf": aligned_wf,
        "trough_to_peak_time": trough_to_peak_time,
        "halfwidth_at_third_max": halfwidth_at_third_max,
    }


def calculate_waveform_halfwidth(waveform: np.ndarray, level: float) -> float:
    """
    Spike width at certain level of the trough amplitude
    adapted from  Rev 1.2, April 2006 (Patrick Egan) Matlab version

    """

    timestamps = np.arange(0, len(waveform))

    # Finde Index des Peaks
    center_index = int(np.argmin(waveform))
    waveform_abs = np.abs(waveform)

    # find first crossing
    i = center_index

    if center_index == len(waveform) - 1:
        return np.nan

    while waveform_abs[i] > level and i > 0:
        i -= 1
    if i == 0:
        return np.nan
    else:
        i = i + 1

    f = interp1d(
        waveform_abs[i - 1 : i + 1], timestamps[i - 1 : i + 1]
    )  # Interpolation
    t_lead = f(level)

    # Second crossing
    i = center_index + 1
    while i < len(waveform_abs) - 1 and waveform_abs[i] > level:
        i += 1

    if i == len(waveform_abs) - 1:
        return np.nan

    t_trail = interp1d(waveform_abs[i - 1 : i + 1], timestamps[i - 1 : i + 1])(
        level
    )  # Interpolation

    return t_trail - t_lead


def calculate_waveform_trough_to_peak_time(waveform: np.ndarray) -> float:
    """adapted from open ephys github repository"""

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform[trough_idx:]) + trough_idx

    time = np.arange(0, 61)
    trough_idx_new = np.interp(trough_idx, time, waveform)
    peak_idx_new = np.interp(peak_idx, time, waveform)

    TP_distance = np.abs(peak_idx_new - trough_idx_new)

    return TP_distance


def calculate_waveform_trough_to_peak_time_new(waveform: np.ndarray) -> float:
    """adapted from open ephys github repository"""

    trough_idx = np.argmin(waveform)
    peak_idx = np.argmax(waveform[trough_idx:]) + trough_idx

    time = np.arange(0, 61)
    TP_distance = np.abs(peak_idx - trough_idx)

    return TP_distance


def get_waveform_metrics(
    directory: str, spiketimes: np.ndarray, spikeclusters: np.ndarray
) -> dict:
    gwfparams = {
        "dataDir": directory,
        "fileName": "temp_wh.dat",
        "spikeTimes": spiketimes,
        "spikeClusters": spikeclusters,
        "nWf": 300,
        "nCh": 384,
        "wfWin": [-20, 41],
        "dataType": np.int16,
    }

    results = get_waveforms(gwfparams)

    # from matplotlib import pyplot as plt

    # plt.figure()
    # plt.plot(results["waveFormsMean_norm"].T)
    # plt.figure()
    # plt.scatter(results["trough_to_peak_time"], results["halfwidth_at_third_max"])
    # plt.show()

    return results


def do_clustering_and_plot(all_data: pd.DataFrame) -> np.ndarray:
    aligned_wf = np.vstack(all_data["aligned_waveforms"])
    cell_type = np.full(len(aligned_wf), "")

    pca_model = PCA(n_components=2)
    pcas_wf = pca_model.fit_transform(aligned_wf)
    lower, upper = np.percentile(pcas_wf[:, 0], [1, 99])
    bad_clusters = np.logical_or(pcas_wf[:, 0] < lower, pcas_wf[:, 0] > upper)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(pcas_wf[:, 0], 100)
    plt.axvline(lower)
    plt.axvline(upper)
    plt.legend(["1%", "99%"])
    plt.subplot(1, 2, 2)
    plt.plot(aligned_wf[bad_clusters].T)
    plt.title("Excluded Clusters")

    # Data (n Samples, d Features)
    data = np.vstack((all_data["New_PT"], all_data["fullwidth_at_third_max"])).T
    data[np.isnan(data)] = 0

    gmm = GaussianMixture(n_components=4, reg_covar=0.001, max_iter=50)
    gmm.fit(data)
    idx = gmm.predict(data)
    cluster0 = data[idx == 0]
    cluster1 = data[idx == 1]
    cluster2 = data[idx == 2]
    cluster3 = data[idx == 3]
    ind_inhibitory = np.argmin(
        [
            sum(np.mean(cluster0, axis=0)),
            sum(np.mean(cluster1, axis=0)),
            sum(np.mean(cluster2, axis=0)),
            sum(np.mean(cluster3, axis=0)),
        ]
    )

    cell_type[idx == ind_inhibitory] = "I"
    cell_type[idx != ind_inhibitory] = "E"
    cell_type[bad_clusters] = "N"

    plt.figure()
    plt.suptitle(all_data["region"].tolist()[0])
    plt.scatter(data[cell_type == "I"][:, 0], data[cell_type == "I"][:, 1], c="red")
    plt.scatter(data[cell_type == "E"][:, 0], data[cell_type == "E"][:, 1], c="blue")
    plt.legend(["Putative inhibitory", "Putative excitatory"])
    plt.figure()
    plt.suptitle(all_data["region"].tolist()[0])
    plt.subplot(1, 2, 1)
    plt.plot(aligned_wf[cell_type == "I"].T)
    plt.plot(
        np.nanmean(
            aligned_wf[cell_type == "I"],
            axis=0,
        ),
        lw=7,
    )
    plt.title("Putative inhibitory (n=" + str(sum(cell_type == "I")) + ")")
    plt.subplot(1, 2, 2)
    plt.plot(aligned_wf[cell_type == "E"].T)
    plt.plot(
        np.nanmean(
            aligned_wf[cell_type == "E"],
            axis=0,
        ),
        lw=7,
    )
    plt.title("Putative excitatory (n=" + str(sum(cell_type == "E")) + ")")

    return cell_type


def main() -> None:
    gwfparams = {
        "dataDir": "C:/Users/jzuen/Documents/Test_data/ks_imec1",
        "fileName": "temp_wh.dat",
        "spikeTimes": np.load(
            "C:/Users/jzuen/Documents/Test_data/ks_imec1/spike_times.npy"
        ),
        "spikeClusters": np.load(
            "C:/Users/jzuen/Documents/Test_data/ks_imec1/spike_clusters.npy"
        ),
        "nWf": 300,
        "nCh": 384,
        "wfWin": [-20, 41],
        "dataType": np.int16,
    }

    results = get_waveforms(gwfparams)

    aligned_wf = results["aligned_wf"]

    from sklearn.decomposition import PCA

    pca_model = PCA(n_components=2)
    pcas_wf = pca_model.fit_transform(aligned_wf)
    lower, upper = np.percentile(pcas_wf[:, 0], [1, 99])
    bad_clusters = np.logical_or(pcas_wf[:, 0] < lower, pcas_wf[:, 0] > upper)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.hist(pcas_wf[:, 0], 100)
    plt.axvline(lower)
    plt.axvline(upper)
    plt.legend(["1%", "99%"])
    plt.subplot(1, 2, 2)
    plt.plot(aligned_wf[bad_clusters].T)
    plt.title("Excluded Clusters")
    plt.show()

    good_cluster = ~bad_clusters
    aligned_wf_sorted = aligned_wf[good_cluster]

    # Data (n Samples, d Features)
    data = np.vstack(
        (results["trough_to_peak_time"], results["halfwidth_at_third_max"])
    ).T
    data[np.isnan(data)] = 0
    data = data[good_cluster]

    # GMM , n_components = number of clusters
    gmm = GaussianMixture(n_components=4, reg_covar=0.001, max_iter=50)
    gmm.fit(data)
    idx = gmm.predict(data)
    cluster0 = data[idx == 0]
    cluster1 = data[idx == 1]
    cluster2 = data[idx == 2]
    cluster3 = data[idx == 3]
    ind_inhibitory = np.argmin(
        [
            sum(np.mean(cluster0, axis=0)),
            sum(np.mean(cluster1, axis=0)),
            sum(np.mean(cluster2, axis=0)),
            sum(np.mean(cluster3, axis=0)),
        ]
    )
    inh_cluster = data[idx == ind_inhibitory]
    exc_cluster = data[idx != ind_inhibitory]

    plt.figure()
    plt.scatter(inh_cluster[:, 0], inh_cluster[:, 1], c="red")
    plt.scatter(exc_cluster[:, 0], exc_cluster[:, 1], c="blue")
    plt.legend(["Putative inhibitory", "Putative excitatory"])
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(aligned_wf_sorted[idx == ind_inhibitory].T)
    plt.plot(
        np.nanmean(
            aligned_wf_sorted[idx == ind_inhibitory],
            axis=0,
        ),
        lw=7,
    )
    plt.title("Putative inhibitory (n=" + str(len(inh_cluster)) + ")")
    plt.subplot(1, 2, 2)
    plt.plot(aligned_wf_sorted[idx != ind_inhibitory].T)
    plt.plot(
        np.nanmean(
            aligned_wf_sorted[idx != ind_inhibitory],
            axis=0,
        ),
        lw=7,
    )
    plt.title("Putative excitatory (n=" + str(len(exc_cluster)) + ")")
    plt.show()

    print("done")
