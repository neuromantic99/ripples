from typing import Tuple
from npyx import extract_rawChunk, read_metadata, get_npix_sync
import numpy as np

from ripples.utils import threshold_detect

"""Put the npyx import into its own file as it confuses pytest"""


def load_lfp_npyx(data_path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    meta = read_metadata(data_path)
    sync = get_npix_sync(data_path, output_binary=True, filt_key="lowpass")
    sampling_rate_lfp = meta["lowpass"]["sampling_rate"]
    # For some reason this is a matrix where the 6th element is the sync signal.
    # check this is the case for all recordings
    assert np.sum(sync[:, 6]) > 0
    assert np.sum(sync) == np.sum(sync[:, 6])
    sync = sync[:, 6]

    sync_idxs = threshold_detect(sync, 0.5)

    inter_pulse_interval_first_five = np.diff(sync_idxs[:4])
    # Check that the first 5 sync pulses are the 5Hz rotary encoder onset
    assert np.all(inter_pulse_interval_first_five < 1000) and np.all(
        inter_pulse_interval_first_five > 100
    )

    lfp = extract_rawChunk(
        data_path,
        [
            0,
            (meta["recording_length_seconds"]),
        ],  # now taking the recording length as a float
        channels=np.arange(384),
        filt_key="lowpass",  # NPX data is devided in "high-pass" = spiking data and "low-pass" = LFP, no filter is being applied
        save=0,
        whiten=0,
        med_sub=False,
        hpfilt=False,
        hpfiltf=0,
        filter_forward=False,
        filter_backward=False,
        nRangeWhiten=None,
        nRangeMedSub=None,
        use_ks_w_matrix=True,
        ignore_ks_chanfilt=True,
        center_chans_on_0=False,
        verbose=False,
        scale=False,
        again=False,
    )

    assert len(sync) == lfp.shape[1] 
    return lfp, sync, sampling_rate_lfp
