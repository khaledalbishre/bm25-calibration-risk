"""
Calibration-specific diagnostics and the constant training-prevalence
baseline for the two-fold calibration of Table 3 (manuscript Table 4).

Estimators: the five of Table 3 (predictions from the ORIGINAL
10_twofold_calibration.predictions(), unchanged) plus `constant`, which
predicts the judged-pair base rate of the collection the calibrators are
fitted on (training-set prevalence; no access to evaluation labels).

Diagnostics (fixed before the analysis was run):
  Brier score and its CORP decomposition (Dimitriadis, Gneiting & Jordan,
  PNAS 2021):  BS = MCB - DSC + UNC, where pc = PAV (isotonic, ties pooled)
  recalibration of the predictions on the evaluation pairs,
      MCB = BS(p) - BS(pc)      miscalibration   (>= 0, lower is better)
      DSC = BS(ybar) - BS(pc)   discrimination   (>= 0, higher is better)
      UNC = BS(ybar) = ybar(1-ybar)               (same for all estimators)
  CITL = mean(p) - ybar (calibration-in-the-large).
  Pooled out-of-fold row: both directions' predictions concatenated
  (10,970 judged pairs), decomposed jointly.

Uncertainty: query-level paired percentile bootstrap with EXACTLY the draws
of 10_twofold_calibration.brier_bootstrap() (same seed, same order,
queries resampled within each evaluation collection, calibrators fixed).
The script first checks that it reproduces the original Brier intervals.

Writes results/revision/
  tab_calibration_diagnostics.csv   point estimates + 95% CIs
  tab_calibration_diag_contrasts.csv paired differences (Brier, MCB, DSC)
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from revision_common import common, m07, m10, OUT, ROOT, DIRECTIONS

EST = ["constant", "norm", "platt", "isotonic", "norm+platt", "norm+isotonic"]
CONTRASTS = [(e, "constant") for e in EST[1:]] + [
    ("norm+platt", "platt"), ("norm+isotonic", "isotonic"),
    ("norm+platt", "norm"), ("norm+isotonic", "norm"), ("platt", "norm"), ("isotonic", "norm")]
LABELS = [f"{tr}->{te}" for tr, te in DIRECTIONS] + ["pooled_oof"]


def corp(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    pc = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y).predict(p)
    yb = y.mean()
    bs = np.mean((p - y) ** 2); bsc = np.mean((pc - y) ** 2); unc = yb * (1 - yb)
    return {"brier": bs, "mcb": bs - bsc, "dsc": unc - bsc, "unc": unc, "citl": p.mean() - yb}


def load_frames():
    runs = {s: m07.load_run(s) for s in ("dl19", "dl20")}
    frames = {}
    for s in ("dl19", "dl20"):
        f = pd.read_parquet(common.DATA_DIR / f"calib_frame_{s}.parquet").copy()
        f["norm_full"] = m07.norm_of_frame(f, runs[s])
        frames[s] = f
    return frames


def main():
    frames = load_frames()
    pred, ys, qidx = {}, {}, {}
    for k, (tr, te) in enumerate(DIRECTIONS):
        test = frames[te]
        pr = m10.predictions(frames[tr], test)
        pr["constant"] = np.full(len(test), frames[tr]["label"].mean())
        for e in EST:
            pred[k, e] = np.asarray(pr[e], float)
        ys[k] = test["label"].to_numpy(float)
        uq, inv = np.unique(test["qid"].to_numpy(), return_inverse=True)
        qidx[k] = [np.where(inv == i)[0] for i in range(len(uq))]

    # identical draws to m10.brier_bootstrap
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    draws = [[rng.integers(0, len(qidx[k]), len(qidx[k])) for k in range(2)]
             for _ in range(common.BOOTSTRAP_RESAMPLES)]

    def stats(sel):
        """sel[k] = pair indices for direction k; returns {(label, est): corp dict}"""
        out = {}
        for e in EST:
            for k in range(2):
                out[LABELS[k], e] = corp(pred[k, e][sel[k]], ys[k][sel[k]])
            out["pooled_oof", e] = corp(np.concatenate([pred[k, e][sel[k]] for k in range(2)]),
                                        np.concatenate([ys[k][sel[k]] for k in range(2)]))
        return out

    full = stats([np.arange(len(ys[k])) for k in range(2)])
    boots = []
    for d in draws:
        sel = [np.concatenate([qidx[k][i] for i in d[k]]) for k in range(2)]
        boots.append(stats(sel))

    # reproduction check against the original Brier intervals (non-constant estimators)
    ref = pd.read_csv(ROOT / "results" / "tab_calibration_brier_ci.csv")
    maxdev = 0.0
    for _, r in ref.iterrows():
        b = np.array([bt[r["direction"], r["estimator"]]["brier"] for bt in boots])
        lo, hi = np.percentile(b, [2.5, 97.5])
        maxdev = max(maxdev, abs(lo - r["ci_lo"]), abs(hi - r["ci_hi"]),
                     abs(full[r["direction"], r["estimator"]]["brier"] - r["brier"]))
    print(f"reproduction check vs results/tab_calibration_brier_ci.csv: max |dev| = {maxdev:.2e}")
    assert maxdev < 1e-12, "bootstrap draws do not reproduce the original intervals"

    rows = []
    for lab in LABELS:
        for e in EST:
            row = {"direction": lab, "estimator": e, **full[lab, e]}
            for m in ("brier", "mcb", "dsc", "citl"):
                lo, hi = np.percentile([bt[lab, e][m] for bt in boots], [2.5, 97.5])
                row[f"{m}_ci_lo"], row[f"{m}_ci_hi"] = lo, hi
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tab_calibration_diagnostics.csv", index=False)

    crow = []
    for lab in LABELS:
        for a, b in CONTRASTS:
            for m in ("brier", "mcb", "dsc"):
                d = np.array([bt[lab, a][m] - bt[lab, b][m] for bt in boots])
                lo, hi = np.percentile(d, [2.5, 97.5])
                crow.append({"direction": lab, "contrast": f"{a} - {b}", "metric": m,
                             "delta": full[lab, a][m] - full[lab, b][m], "ci_lo": lo, "ci_hi": hi,
                             "ci_excludes_0": bool(lo > 0 or hi < 0)})
    C = pd.DataFrame(crow)
    C.to_csv(OUT / "tab_calibration_diag_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(T[["direction", "estimator", "brier", "brier_ci_lo", "brier_ci_hi", "mcb", "mcb_ci_lo", "mcb_ci_hi",
             "dsc", "dsc_ci_lo", "dsc_ci_hi", "unc", "citl"]].round(4).to_string(index=False))
    print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
