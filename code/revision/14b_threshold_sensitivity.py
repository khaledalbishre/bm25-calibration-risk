"""
Absolute nDCG@10 thresholds, step 2 (manuscript Appendix A.3, Table A3):
robustness of the RQ3 comparison to absolute targets. Thresholds fixed
before this script was written (feasibility rule of 14a):
primary T1=0.35, T2=0.50; supplementary sweep 0.35..0.65 (all feasible).
Writes results/revision/tab_threshold_auroc.csv, tab_threshold_contrasts.csv
"""
import numpy as np
import pandas as pd
from revision_common import m07, OUT
from revision_signals import load_inputs, build_P, original_boots

PRIMARY = (0.35, 0.50)
SWEEP = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
SIGNALS = [("cv100", "CV@100"), ("top1", "Raw top-1"), ("mean10", "Raw mean@10"),
           ("qrcmean_norm", "QRC-mean norm"), ("qrcmean_platt", "QRC-mean Platt"),
           ("qrcmean_iso", "QRC-mean isotonic"), ("qrcmean_nplatt", "QRC-mean norm+Platt"),
           ("qrcmean_niso", "QRC-mean norm+isotonic"), ("qrcmax_platt", "QRC-max Platt"),
           ("qrcmax_iso", "QRC-max isotonic")]
CONTRASTS = [("qrcmean_platt", "mean10"), ("qrcmean_nplatt", "qrcmean_platt"),
             ("qrcmean_niso", "qrcmean_iso"), ("qrcmean_nplatt", "qrcmean_norm")]


def main():
    runs, frames, ndcg10 = load_inputs()
    P = build_P(runs, frames, ndcg10)
    folds = P["fold"].to_numpy()
    boots = original_boots(folds)
    rows, crow = [], []
    for t in SWEEP:
        y = (P["ndcg10"].to_numpy() < t).astype(int)
        B = {c: np.array([m07.stratified_auc(y[i], P[c].to_numpy(float)[i], folds[i]) for i in boots])
             for c, _ in SIGNALS}
        pt = {c: m07.stratified_auc(y, P[c].to_numpy(float), folds) for c, _ in SIGNALS}
        for c, lab in SIGNALS:
            lo, hi = np.percentile(B[c], [2.5, 97.5]); dlo, dhi = np.percentile(B[c] - B["cv100"], [2.5, 97.5])
            rows.append({"threshold": t, "primary": t in PRIMARY, "signal": lab,
                         "pos_dl19": int(y[folds == "dl19"].sum()), "pos_dl20": int(y[folds == "dl20"].sum()),
                         "auroc_stratified": pt[c], "ci_lo": lo, "ci_hi": hi,
                         "delta_vs_cv100": pt[c] - pt["cv100"], "delta_ci_lo": dlo, "delta_ci_hi": dhi,
                         "delta_ci_excludes_0": bool(dlo > 0 or dhi < 0)})
        for a, b in CONTRASTS:
            lo, hi = np.percentile(B[a] - B[b], [2.5, 97.5])
            crow.append({"threshold": t, "primary": t in PRIMARY, "contrast": f"{a} - {b}",
                         "delta_auroc": pt[a] - pt[b], "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    T = pd.DataFrame(rows); C = pd.DataFrame(crow)
    T.to_csv(OUT / "tab_threshold_auroc.csv", index=False); C.to_csv(OUT / "tab_threshold_contrasts.csv", index=False)
    pd.set_option("display.width", 220)
    print(T[T.primary].round(3).to_string(index=False)); print()
    print(C[C.primary].round(3).to_string(index=False)); print()
    print(T.pivot(index="signal", columns="threshold", values="auroc_stratified").round(3).to_string())


if __name__ == "__main__":
    main()
