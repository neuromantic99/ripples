import csv
import numpy as np
import mat73
import os
import pandas as pd
from pathlib import Path
import matplotlib
from scipy.signal import find_peaks
from scipy.io import loadmat

from utils import split_vector_consecutive


# Need to convert this manually because can't load matlab tables into python
overview_table = pd.read_csv("overview.csv")

GENOTYPE = overview_table["Genotype"]
TRAJECTORY = overview_table["TrajectoryPrefix"]
EXP_DATE = overview_table["AnimalID"]
TIMEPOINT = overview_table["Timepoint"]
DATA_DIR = Path("/Volumes/MarcBusche/Jana/Neuropixels/Trajectories/Processed data")
PROBE_NUM = overview_table["Imec_1_Traj"]


# Load channel map
# Assuming the channel map file is loaded similarly
chan_map = mat73.loadmat(
    "/Volumes/MarcBusche/Matlab Code/NPX Config/long_linear_shank_ChanMapBank01.mat"
)

swr_vits = {
    "depth_axis": np.arange(20, 7681, 20),
    "swr_freq_range": [80, 250],
    "supra_freq_range": [250, 500],
    "swr_max_length": 0.4,
    "swr_cluster_gap": 0.12,
    "medLevel": 2,
    "detect_type": "Dupret",
    "do_multichan": "Yes",
    "onset_offset": 0.5,
    "restTimeWindow": 6,
    "CARtype": "Original",
}

animal = 0
imec_probe = 1
# I think this is right based on the matlab code
experiment_id = 3

experiment_path = DATA_DIR / Path(
    GENOTYPE[animal] + TRAJECTORY[animal] + EXP_DATE[animal] + TIMEPOINT[animal]
)


# Load probe trajectory details
probe_details = mat73.loadmat(
    experiment_path / f"probe_trajectory_details_0603_imec{imec_probe}.mat"
)

# Load in bandpower resting state data

# '/Volumes/MarcBusche/Jana/Neuropixels/Trajectories/Processed data/NLGF_A_1393311_3M/Baseline3'

baseline_path = experiment_path / f"Baseline{experiment_id}"

freq_data = mat73.loadmat(
    baseline_path / f"lfp_frequency_bands_resting_imec{imec_probe}.mat"
)

SWR_pow = freq_data["freq"]["SWR_pow_rest"]
SRP_pow = freq_data["freq"]["SRP_pow_rest"]

idx_CA1 = [
    i
    for i, region in enumerate(probe_details["probe_details"]["alignedregions"])
    if "Field CA1" in region
]

# Why do you need to do this?
CAchans = 384 - np.array(idx_CA1)

SWR_pow_CA1 = SWR_pow[CAchans]
max_power_channel = np.argmax(SWR_pow_CA1)

maxSWRchan = CAchans[max_power_channel]
SWR_chans = CAchans[(max_power_channel - 2) : (max_power_channel + 3)]

# Load LFP data and synchronization channel data
lfp_data = mat73.loadmat(baseline_path / f"lfp_data_CAR_fin_imec{imec_probe}.mat")
lfp_vits = lfp_data["lfp_vits"]
sync_channel_data = mat73.loadmat(
    baseline_path / f"synch_channel_data_AP_LP_imec{imec_probe}.mat"
)

sync_dat_LF = sync_channel_data["sync_dat_LF"]
# Finding local maxima for synchronization points
peaks, _ = find_peaks(sync_dat_LF, height=None)  # Assuming sync_dat_LF is 1D
onset_syncLF = peaks[0]
offset_syncLF = peaks[-5]  # Adjusted to end-4 to match MATLAB's end-4

# Truncate LFP data to match behavior recording range
dataArray_lfp = lfp_data["dataArray_lfp"][:, onset_syncLF : offset_syncLF + 1]
lfp_tim = np.arange(dataArray_lfp.shape[1]) / lfp_vits["sampling_rate"]

# Replace NaNs (bad/ignored channels) with 0
dataArray_lfp[np.isnan(np.sum(dataArray_lfp, axis=1)), :] = 0

# Optional: filter for 50Hz noise if required
# dataArray_lfp = avgXnpxMainsFilt(dataArray_lfp, axis=1, lfp_vits, notch=50)  # Assuming implementation exists

# Load behavioural data
# restTS is resaved via matlab as I can't load the original .mat
restTS = loadmat("restTS.mat")["resty"]

# minus 1 as python is zero indexed
resting_data = restTS[imec_probe - 1, 0].squeeze()  #

locomotion_period = np.where(resting_data == 0)[0]

# Mask out time-points in LFP data and time vector corresponding to locomotion
print("Masking out movement/locomotion in LFP data")
restLFP = np.array(dataArray_lfp, copy=True)
restLFP[:, locomotion_period] = np.NaN
restLFPtim = np.array(lfp_tim, copy=True)
restLFPtim[locomotion_period] = np.NaN

resting_timepoints = np.arange(len(resting_data))[np.where(resting_data)]
resting_periods_pre = split_vector_consecutive(resting_timepoints)
length_RestingPeriods = [len(period) for period in resting_periods_pre]

idxLongRest = [
    i
    for i, length in enumerate(length_RestingPeriods)
    if length >= swr_vits["restTimeWindow"] * lfp_vits["sampling_rate"]
]

resting_periods = np.concatenate([resting_periods_pre[i] for i in idxLongRest])

rest_time = len(resting_periods) / lfp_vits["sampling_rate"]
swr_dat_restTimeBehaviour = len(resting_timepoints) / lfp_vits["sampling_rate"]


1 / 0
