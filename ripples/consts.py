from pathlib import Path

HERE = Path(__file__).parent
UMBRELLA = Path("//128.40.224.64/marcbusche/Jana/Neuropixels")
RESULTS_PATH = UMBRELLA / "Trajectories" / "results_python" / "0304_6M_5SD_new_bandpass"
# RESULTS_PATH = (
#     HERE.parent / "results" / "0104_5SD_6M"
# )  # "6M_cohort_5SD_1103" cohort_5med_1703# "0104_5SD_6M" "0304_6M_5SD_new_bandpass"
DETECTION_METHOD = "sd"  # options 'median' or 'sd'

RIPPLE_BAND = [110, 200]
SUPRA_RIPPLE_BAND = [250, 500]
