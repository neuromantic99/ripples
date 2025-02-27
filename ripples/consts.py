from pathlib import Path

HERE = Path(__file__).parent
RESULTS_PATH = HERE.parent / "results" / "2102_bigger_cohort_5SD"
DETECTION_METHOD = "sd"  # options 'median' or 'sd'

RIPPLE_BAND = [120, 250]
SUPRA_RIPPLE_BAND = [250, 500]
