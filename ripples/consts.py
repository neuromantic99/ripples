from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE.parent / "results" / "2802_with_good_unit_detection"
DETECTION_METHOD = "median"  # options 'median' or 'sd'

RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]
