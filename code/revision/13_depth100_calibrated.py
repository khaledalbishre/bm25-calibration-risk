"""
Calibrated probability aggregates at depth 100 for the below-median target
(manuscript Appendix A.2, Table A2) and the calibration on/off contrasts of
Table 7. Signals and contrasts were fixed before the analysis was run.
Fixed calibrators and the ORIGINAL bootstrap indices of 07 (paired with
Tables 5-7). Reproduction of results/tab_risk_foldaware.csv is checked first.
Writes results/revision/tab_depth100_signals.csv, tab_depth100_contrasts.csv
"""
import numpy as np
import pandas as pd
from revision_common import m07, OUT, ROOT
from revision_signals import load_inputs, build_P, original_boots, EST

LAB = {"norm": "norm", "platt": "Platt", "iso": "isotonic", "nplatt": "norm+Platt", "niso": "norm+isotonic"}


def main():
    runs, frames, ndcg10 = load_inputs()
    P = build_P(runs, frames, ndcg10)
    folds, y = P["fold"].to_numpy(), P["risk"].to_numpy()
    boots = original_boots(folds)
    sig = lambda c: P[c].to_numpy(float)
    cache = {}

    def bt(c):
        if c not in cache:
            s = sig(c)
            cache[c] = np.array([m07.stratified_auc(y[i], s[i], folds[i]) for i in boots])
        return cache[c]

    # reproduction check
    ref = pd.read_csv(ROOT / "results" / "tab_risk_foldaware.csv")
    dev = 0.0
    for _, r in ref.iterrows():
        b = bt(r["signal"]); lo, hi = np.percentile(b, [2.5, 97.5])
        dev = max(dev, abs(m07.stratified_auc(y, sig(r["signal"]), folds) - r["auroc_stratified"]),
                  abs(lo - r["auroc_ci_lo"]), abs(hi - r["auroc_ci_hi"]))
    print(f"reproduction check vs results/tab_risk_foldaware.csv: max |dev| = {dev:.2e}")
    assert dev < 1e-12

    new = [(f"qrcmean100_{e}", f"QRC-mean@100, {LAB[e]}") for e in EST] + \
          [(f"pstd100_{e}", f"std of P(Rel)@100, {LAB[e]}") for e in EST]
    ref_rows = [("cv100", "CV@100"), ("std100", "std@100"), ("mean100", "mean@100")] + \
               [(f"qrcmean_{e}", f"QRC-mean@10, {LAB[e]}") for e in EST]
    base = bt("cv100"); bpt = m07.stratified_auc(y, sig("cv100"), folds)
    rows = []
    for c, lab in ref_rows + new:
        s = sig(c); pt = m07.stratified_auc(y, s, folds)
        lo, hi = np.percentile(bt(c), [2.5, 97.5]); dlo, dhi = np.percentile(bt(c) - base, [2.5, 97.5])
        rows.append({"signal": c, "label": lab, "depth100_calibrated": (c, lab) in new,
                     "auroc_dl19": m07.risk_auc(y[folds == "dl19"], s[folds == "dl19"]),
                     "auroc_dl20": m07.risk_auc(y[folds == "dl20"], s[folds == "dl20"]),
                     "auroc_stratified": pt, "ci_lo": lo, "ci_hi": hi,
                     "delta_vs_cv100": pt - bpt, "delta_ci_lo": dlo, "delta_ci_hi": dhi,
                     "delta_ci_excludes_0": bool(dlo > 0 or dhi < 0),
                     "n_distinct_dl19": len(np.unique(np.round(s[folds == "dl19"], 12))),
                     "n_distinct_dl20": len(np.unique(np.round(s[folds == "dl20"], 12)))})
    T = pd.DataFrame(rows); T.to_csv(OUT / "tab_depth100_signals.csv", index=False)

    C = []
    def con(tag, a, b):
        d = m07.stratified_auc(y, sig(a), folds) - m07.stratified_auc(y, sig(b), folds)
        lo, hi = np.percentile(bt(a) - bt(b), [2.5, 97.5])
        C.append({"family": tag, "a": a, "b": b, "delta_auroc": d, "ci_lo": lo, "ci_hi": hi,
                  "ci_excludes_0": bool(lo > 0 or hi < 0)})
    for e in EST:
        con("E-a depth (calibrated level)", f"qrcmean100_{e}", f"qrcmean_{e}")
    for e in EST:
        con("E-b QRC-mean@100 vs CV@100", f"qrcmean100_{e}", "cv100")
    for e in EST:
        con("E-c std P@100 vs CV@100", f"pstd100_{e}", "cv100")
    for e in ("platt", "iso"):
        con("E-d dispersion vs level @100 (calibrated)", f"pstd100_{e}", f"qrcmean100_{e}")
    con("E-e calibration effect @100", "qrcmean100_platt", "mean100")
    con("E-e calibration effect @100", "qrcmean100_nplatt", "qrcmean100_platt")
    con("E-e calibration effect @100", "qrcmean100_niso", "qrcmean100_iso")
    for k in ("", "100"):
        con(f"E-f calibration on vs off (same input) @{k or '10'}", f"qrcmean{k}_nplatt", f"qrcmean{k}_norm")
        con(f"E-f calibration on vs off (same input) @{k or '10'}", f"qrcmean{k}_niso", f"qrcmean{k}_norm")
    C = pd.DataFrame(C); C.to_csv(OUT / "tab_depth100_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.drop(columns=["signal"]).round(3).to_string(index=False)); print()
    print(C.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
