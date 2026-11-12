"""
Renders manuscript Table 4, the two calibration on/off rows of Table 7,
and Appendix Tables A1-A3 as Markdown directly from the result tables in
results/revision/, so every printed number is generated, not typed.
Writes results/revision/revision_tables.md
"""
import pandas as pd
from revision_common import OUT

f3 = lambda x: f"{x:.3f}".replace("-0.000", "0.000")
def sg(x):
    s = f3(x)
    return s if s.startswith("-") else ("+" + s if x > 0 and s != "0.000" else s)
def ci(lo, hi): return f"[{f3(lo)}, {f3(hi)}]".replace("-", "−")
f4 = lambda x: f"{x:.4f}".replace("-0.0000", "0.0000")
def sg4(x):
    s = f4(x)
    return s if s.startswith("-") else ("+" + s if x > 0 and s != "0.0000" else s)
def ci4(lo, hi): return f"[{f4(lo)}, {f4(hi)}]".replace("-", "−")
def m(s): return s.replace("-", "−")

def main():
    out = []
    D = pd.read_csv(OUT / "tab_calibration_diagnostics.csv").set_index(["direction", "estimator"])
    names = [("constant", "constant (training prevalence)"), ("norm", "norm"), ("platt", "Platt"),
             ("isotonic", "isotonic"), ("norm+platt", "norm+Platt"), ("norm+isotonic", "norm+isotonic")]
    out.append("TABLE 4\n| Estimator | Brier (19→20) | MCB (19→20) | Brier (20→19) | MCB (20→19) | Pooled Brier | Pooled MCB [95% CI] | Pooled DSC |\n|---|---|---|---|---|---|---|---|")
    for k, lab in names:
        a, b, p = D.loc["dl19->dl20", k], D.loc["dl20->dl19", k], D.loc["pooled_oof", k]
        out.append(f"| {lab} | {f3(a.brier)} | {f3(a.mcb)} | {f3(b.brier)} | {f3(b.mcb)} | {f3(p.brier)} | "
                   f"{f3(p.mcb)} {ci(p.mcb_ci_lo, p.mcb_ci_hi)} | {f3(p.dsc)} |")
    out.append(f"UNC: 19->20 {f3(D.loc['dl19->dl20','norm'].unc)}, 20->19 {f3(D.loc['dl20->dl19','norm'].unc)}, pooled {f3(D.loc['pooled_oof','norm'].unc)}")

    C = pd.read_csv(OUT / "tab_depth100_contrasts.csv")
    out.append("\nTABLE 7 NEW ROWS")
    lab = {"qrcmean_nplatt": "QRC-mean (norm+Platt) − QRC-mean (norm)", "qrcmean_niso": "QRC-mean (norm+isotonic) − QRC-mean (norm)"}
    for r in C[C.family.str.startswith("E-f") & C.family.str.endswith("@10")].itertuples():
        out.append(f"| {lab[r.a]} | calibration on vs. off, same input | {m(sg(r.delta_auroc))} {ci(r.ci_lo, r.ci_hi)} |")

    # A1 refit
    out.append("\nTABLE A1\n| Quantity (pooled unless stated) | Point estimate | 95% CI, calibrators fixed | 95% CI, calibrators refitted | Interval excludes 0 (fixed / refitted) |\n|---|---|---|---|---|")
    B = pd.read_csv(OUT / "tab_refit_brier_contrasts.csv")
    M = pd.read_csv(OUT / "tab_refit_mcb_contrasts.csv")
    A = pd.read_csv(OUT / "tab_refit_auroc_contrasts.csv")
    yn = lambda a, b: f"{'yes' if a else 'no'} / {'yes' if b else 'no'}"
    bl = {"nplatt": "norm+Platt", "niso": "norm+isotonic", "platt": "Platt", "iso": "isotonic", "norm": "norm", "constant": "constant"}
    for d, a, b in [("pooled_oof", "nplatt", "platt"), ("pooled_oof", "niso", "iso"), ("pooled_oof", "nplatt", "norm"),
                    ("pooled_oof", "platt", "constant"), ("pooled_oof", "nplatt", "constant"),
                    ("dl19->dl20", "platt", "norm"), ("dl19->dl20", "nplatt", "norm")]:
        r = B[B.contrast == f"{d}: {a} - {b}"].iloc[0]
        dd = "" if d == "pooled_oof" else " (19→20)"
        out.append(f"| Brier, {bl[a]} − {bl[b]}{dd} | {m(sg4(r.point))} | {ci4(r.fixed_lo, r.fixed_hi)} | {ci4(r.refit_lo, r.refit_hi)} | {yn(r.fixed_excl0, r.refit_excl0)} |")
    ml = {"norm+platt": "norm+Platt", "norm+isotonic": "norm+isotonic", "platt": "Platt", "isotonic": "isotonic", "norm": "norm", "constant": "constant"}
    for r in M[M.direction == "pooled_oof"].itertuples():
        a, b = r.contrast.split(" - ")
        out.append(f"| MCB, {ml[a]} − {ml[b]} | {m(sg4(r.delta_mcb))} | {ci4(r.fixed_lo, r.fixed_hi)} | {ci4(r.refit_lo, r.refit_hi)} | {yn(r.fixed_excl0, r.refit_excl0)} |")
    al = {"qrcmean_platt": "QRC-mean (Platt)", "mean10": "mean@10", "qrcmean_nplatt": "QRC-mean (norm+Platt)",
          "qrcmean_niso": "QRC-mean (norm+isotonic)", "qrcmean_iso": "QRC-mean (isotonic)", "qrcmean_norm": "QRC-mean (norm)",
          "cv100": "CV@100", "qrcmax_platt": "QRC-max (Platt)", "qrcmax_iso": "QRC-max (isotonic)"}
    keep = ["qrcmean_platt - mean10", "qrcmean_nplatt - qrcmean_platt", "qrcmean_niso - qrcmean_iso",
            "qrcmean_nplatt - qrcmean_norm", "qrcmean_platt - cv100", "qrcmean_iso - cv100", "qrcmean_norm - cv100",
            "qrcmean_nplatt - cv100", "qrcmean_niso - cv100", "qrcmax_platt - cv100", "qrcmax_iso - cv100"]
    for k in keep:
        r = A[A.contrast == k].iloc[0]; a, b = k.split(" - ")
        out.append(f"| AUROC, {al[a]} − {al[b]} | {m(sg(r.point))} | {ci(r.fixed_lo, r.fixed_hi)} | {ci(r.refit_lo, r.refit_hi)} | {yn(r.fixed_excl0, r.refit_excl0)} |")

    # A2 depth 100
    S = pd.read_csv(OUT / "tab_depth100_signals.csv")
    out.append("\nTABLE A2\n| Signal | AUROC DL19 | AUROC DL20 | AUROC stratified [95% CI] | Δ vs. CV@100 [95% CI] |\n|---|---|---|---|---|")
    for r in S[S.depth100_calibrated | (S.signal == "cv100")].itertuples():
        lab = r.label.replace("P(Rel)", "P(Rel)")
        d = "—" if r.signal == "cv100" else (("**" if r.delta_ci_excludes_0 else "") + f"{m(sg(r.delta_vs_cv100))} {ci(r.delta_ci_lo, r.delta_ci_hi)}" + ("**" if r.delta_ci_excludes_0 else ""))
        out.append(f"| {lab} | {f3(r.auroc_dl19)} | {f3(r.auroc_dl20)} | {f3(r.auroc_stratified)} {ci(r.ci_lo, r.ci_hi)} | {d} |")
    out.append("A2 contrasts:")
    for r in C.itertuples():
        out.append(f"  {r.family}: {r.a} - {r.b} = {sg(r.delta_auroc)} {ci(r.ci_lo, r.ci_hi)} excl0={r.ci_excludes_0}")

    # A3 threshold sweep
    T = pd.read_csv(OUT / "tab_threshold_auroc.csv")
    ths = sorted(T.threshold.unique())
    out.append("\nTABLE A3\n| Signal | " + " | ".join(f"< {t:.2f}" for t in ths) + " |\n|" + "---|" * (len(ths) + 1))
    cnt = T.drop_duplicates("threshold").set_index("threshold")
    out.append("| Positive queries (DL19 / DL20) | " + " | ".join(f"{int(cnt.loc[t,'pos_dl19'])} / {int(cnt.loc[t,'pos_dl20'])}" for t in ths) + " |")
    order = ["CV@100", "Raw top-1", "Raw mean@10", "QRC-mean norm", "QRC-mean Platt", "QRC-mean isotonic",
             "QRC-mean norm+Platt", "QRC-mean norm+isotonic", "QRC-max Platt", "QRC-max isotonic"]
    for sname in order:
        cells = []
        for t in ths:
            r = T[(T.threshold == t) & (T.signal == sname)].iloc[0]
            v = f3(r.auroc_stratified)
            cells.append(f"**{v}**" if r.delta_ci_excludes_0 else v)
        out.append(f"| {sname.replace('QRC-mean ', 'QRC-mean, ').replace('QRC-max ', 'QRC-max, ')} | " + " | ".join(cells) + " |")
    out.append("PRIMARY DETAIL:")
    for r in T[T.primary].itertuples():
        out.append(f"  t={r.threshold} {r.signal}: {f3(r.auroc_stratified)} {ci(r.ci_lo, r.ci_hi)} d={sg(r.delta_vs_cv100)} {ci(r.delta_ci_lo, r.delta_ci_hi)} excl={r.delta_ci_excludes_0}")
    TC = pd.read_csv(OUT / "tab_threshold_contrasts.csv")
    for r in TC.itertuples():
        out.append(f"  t={r.threshold} {r.contrast}: {sg(r.delta_auroc)} {ci(r.ci_lo, r.ci_hi)} excl={r.ci_excludes_0}")
    (OUT / "revision_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))

if __name__ == "__main__":
    main()
