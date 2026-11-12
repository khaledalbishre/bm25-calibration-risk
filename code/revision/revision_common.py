"""
Shared loader for the supplementary robustness analyses in code/revision/.

Imports the original pipeline modules from code/ unchanged and writes only
to results/revision/. Run the scripts from code/revision/ (or from the
repository root with `python code/revision/<script>.py`).
"""
import sys
from importlib import import_module
from pathlib import Path

REV_CODE = Path(__file__).resolve().parent                # code/revision/
ROOT = REV_CODE.parent.parent                             # repository root
ORIG_CODE = ROOT / "code"
OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REV_CODE))
sys.path.insert(0, str(ORIG_CODE))
common = import_module("00_common")
m04 = import_module("04_fit_calibrators")
m05 = import_module("05_risk_signal")
m07 = import_module("07_fold_aware_risk")
m10 = import_module("10_twofold_calibration")

FOLDS = ("dl19", "dl20")
DIRECTIONS = (("dl19", "dl20"), ("dl20", "dl19"))   # (fit on, evaluate on)
