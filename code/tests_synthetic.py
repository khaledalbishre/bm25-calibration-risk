"""
Self-tests for the pure-math helpers in 04_fit_calibrators.py and
05_risk_signal.py, run against small hand-built synthetic arrays with
known-correct answers. These do NOT touch MS MARCO / TREC DL data and
produce no number usable in the paper — they only check that the
calibration/ECE/Brier/AUROC/bootstrap code is implemented correctly, so
that once real data is available Phases 1-4
can run unattended with no further debugging.

Usage: python code/tests_synthetic.py
"""
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

m04 = import_module("04_fit_calibrators")
m05 = import_module("05_risk_signal")

failures = []


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    if not cond:
        failures.append(name)


# ---- ece ----
# Perfectly calibrated: prob == empirical frequency in every bin -> ECE = 0
probs = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                   0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 1/10 = 0.1 matches p=0.1
                    1, 1, 1, 1, 1, 1, 1, 1, 0, 0])  # 8/10 = 0.8 vs p=0.9 -> ECE contribution
e10 = m04.ece(probs, labels, 10)
expected = 0.5 * abs(0.1 - 0.1) + 0.5 * abs(0.8 - 0.9)
check("ece: matches hand-computed value", abs(e10 - expected) < 1e-9)

# All predictions correct and extreme -> ECE = 0
probs2 = np.array([0.0, 0.0, 1.0, 1.0])
labels2 = np.array([0, 0, 1, 1])
check("ece: perfect predictions give ECE=0", abs(m04.ece(probs2, labels2, 10)) < 1e-9)

# ---- brier ----
b = m04.brier(np.array([1.0, 0.0, 0.5]), np.array([1, 0, 1]))
expected_b = np.mean([(1 - 1) ** 2, (0 - 0) ** 2, (0.5 - 1) ** 2])
check("brier: matches hand-computed value", abs(b - expected_b) < 1e-9)

# ---- norm_transform ----
df = pd.DataFrame({"qid": ["a", "a", "a", "b", "b"], "bm25_score": [1.0, 2.0, 3.0, 5.0, 5.0]})
out = m04.norm_transform(df)
check("norm_transform: min->0, max->1 within query a", out[0] == 0.0 and out[2] == 1.0)
check("norm_transform: mid value = 0.5 within query a", abs(out[1] - 0.5) < 1e-9)
check("norm_transform: degenerate tie query -> 0.5", out[3] == 0.5 and out[4] == 0.5)

# ---- assert_ranking_invariant ----
ok_df = pd.DataFrame({
    "qid": ["a", "a", "a"],
    "bm25_score": [3.0, 2.0, 1.0],
    "cal": [0.9, 0.5, 0.5],  # non-increasing as score decreases -> OK (isotonic-style tie)
})
try:
    m04.assert_ranking_invariant(ok_df, "bm25_score", "cal")
    check("assert_ranking_invariant: accepts monotone-with-ties", True)
except AssertionError:
    check("assert_ranking_invariant: accepts monotone-with-ties", False)

bad_df = pd.DataFrame({
    "qid": ["a", "a", "a"],
    "bm25_score": [3.0, 2.0, 1.0],
    "cal": [0.5, 0.9, 0.1],  # score rank 1 has LOWER calibrated prob than rank 2 -> violation
})
try:
    m04.assert_ranking_invariant(bad_df, "bm25_score", "cal")
    check("assert_ranking_invariant: rejects a real violation", False)
except AssertionError:
    check("assert_ranking_invariant: rejects a real violation", True)

# ---- nqc ----
check("nqc: std=0 (all-equal top-k) -> 0.0", m05.nqc(np.array([2.0, 2.0, 2.0])) == 0.0)
vals = np.array([1.0, 2.0, 3.0, 4.0])
expected_nqc = vals.std(ddof=0) / vals.mean()
check("nqc: matches hand-computed coefficient of variation", abs(m05.nqc(vals) - expected_nqc) < 1e-9)

# ---- qrc ----
check("qrc: mean aggregation", abs(m05.qrc(np.array([0.2, 0.4, 0.6]), "mean") - 0.4) < 1e-9)
check("qrc: max aggregation", m05.qrc(np.array([0.2, 0.4, 0.6]), "max") == 0.6)

# ---- risk_auroc sign convention ----
# Signal perfectly separates: high signal (confidence) -> label 0 (not risky),
# low signal -> label 1 (risky). This should give AUROC = 1.0 with the
# -signal convention used for risk detection.
labels_r = np.array([1, 1, 0, 0])
signal_r = np.array([0.1, 0.2, 0.8, 0.9])  # low signal <-> risky (label 1)
check("risk_auroc: perfect separation with intended sign -> AUROC=1.0", abs(m05.risk_auroc(labels_r, signal_r) - 1.0) < 1e-9)

# Inverted case: high signal <-> risky -> should give AUROC close to 0.0 (worse than chance)
signal_inv = np.array([0.9, 0.8, 0.2, 0.1])
check("risk_auroc: inverted relationship -> AUROC=0.0", abs(m05.risk_auroc(labels_r, signal_inv) - 0.0) < 1e-9)

# ---- bootstrap_ci ----
rng = np.random.default_rng(0)
n = 100
labels_b = (rng.random(n) < 0.5).astype(int)
# signal_a strongly informative, signal_b pure noise
signal_a = np.where(labels_b == 1, rng.normal(0, 1, n), rng.normal(3, 1, n))
signal_b = rng.normal(0, 1, n)
res = m05.bootstrap_ci(signal_a, signal_b, labels_b, n_resamples=200, seed=1)
check("bootstrap_ci: informative signal_a has AUROC well above 0.5", res["auroc_a"] > 0.8)
check("bootstrap_ci: noise signal_b has AUROC near 0.5", 0.3 < res["auroc_b"] < 0.7)
check("bootstrap_ci: delta CI does not straddle a huge implausible range", -1.0 <= res["delta_ci"][0] <= res["delta_ci"][1] <= 1.0)
check("bootstrap_ci: point delta within its own CI", res["delta_ci"][0] <= res["delta"] <= res["delta_ci"][1])

# ---- fold-aware AUROC (07_fold_aware_risk.py) ----
m07 = import_module("07_fold_aware_risk")
lab_f = np.array([1, 1, 0, 0, 1, 1, 0, 0])
fold_f = np.array(["a"] * 4 + ["b"] * 4)
sig_f = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4])
check("risk_auc: equals sklearn-based risk_auroc", abs(m07.risk_auc(lab_f, sig_f) - m05.risk_auroc(lab_f, sig_f)) < 1e-12)
# Fold-specific level offset: perfect within each fold, but fold b shifted far up.
sig_off = np.where(fold_f == "b", sig_f + 10.0, sig_f)
check("stratified_auc: invariant to fold-specific offsets (=1.0)", abs(m07.stratified_auc(lab_f, sig_off, fold_f) - 1.0) < 1e-12)
check("risk_auc: unstratified pooling is hurt by the same offset (<1.0)", m07.risk_auc(lab_f, sig_off) < 1.0)
wr = m07.within_fold_rank(sig_off, fold_f)
check("within_fold_rank: ranks computed per fold", np.allclose(wr[:4], wr[4:]))

print()
if failures:
    print(f"{len(failures)} TEST(S) FAILED: {failures}")
    sys.exit(1)
else:
    print("ALL SYNTHETIC TESTS PASSED.")
