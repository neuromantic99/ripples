from npyx import extract_rawChunk, read_metadata
import numpy as np

"""Put the npyx import into its own file as it confuses pytest"""


def load_lfp_npyx(data_path: str) -> np.ndarray:
    meta = read_metadata(data_path)
    return extract_rawChunk(
        data_path,
        [0, int(meta["recording_length_seconds"])],
        channels=np.arange(384),
        filt_key="lowpass",
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
