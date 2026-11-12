"""
CORP miscalibration (MCB) under calibrator refitting (manuscript
Appendix A.1, Table A1). Same resamples and refitting as 15_refit_bootstrap.py
(original 07 bootstrap indices); MCB defined exactly as in
12_calibration_diagnostics.corp(). Fixed and refit intervals use identical
query resamples, so they differ only by refitting.
Writes results/revision/tab_refit_mcb.csv, tab_refit_mcb_contrasts.csv
"""
import importlib
import numpy as np
import pandas as pd
from revision_common import m07, OUT, ROOT, DIRECTIONS
from revision_signals import load_inputs, build_P, original_boots

m12 = importlib.import_module("12_calibration_diagnostics")
m15 = importlib.import_module("15_refit_bootstrap")
EST = ["constant", "norm", "platt", "iso", "nplatt", "niso"]
NAME = {"constant": "constant", "norm": "norm", "platt": "platt", "iso": "isotonic",
        "nplatt": "norm+platt", "niso": "norm+isotonic"}
CON = [("nplatt", "platt"), ("niso", "iso"), ("nplatt", "norm"), ("niso", "norm"),
       ("nplatt", "constant"), ("niso", "constant")]
LABELS = [f"{a}->{b}" for a, b in DIRECTIONS] + ["pooled_oof"]


def main():
    runs, frames, ndcg10 = load_inputs()
    P = build_P(runs, frames, ndcg10)
    folds, qids = P["fold"].to_numpy(), P.index.to_numpy()
    boots = original_boots(folds)
    F, pairs = {}, {}
    for s in ("dl19", "dl20"):
        f = frames[s].copy(); f["norm_full"] = m07.norm_of_frame(f, runs[s]); F[s] = f
        q = f["qid"].to_numpy()
        pairs[s] = {k: np.where(q == k)[0] for k in P.index[folds == s]}

    def calibrators(sel):
        out = {}
        for s in ("dl19", "dl20"):
            idx = np.concatenate([pairs[s][k] for k in sel[s]])
            out[s] = m15.fit(F[s]["bm25_score"].to_numpy()[idx], F[s]["norm_full"].to_numpy()[idx],
                             F[s]["label"].to_numpy()[idx])
        return out

    def mcb(cals, sel):
        res, pool = {}, {e: ([], []) for e in EST}
        for k, (tr, te) in enumerate(DIRECTIONS):
            idx = np.concatenate([pairs[te][q] for q in sel[te]])
            raw, nrm, y = (F[te][c].to_numpy()[idx] for c in ("bm25_score", "norm_full", "label"))
            for e in EST:
                p = m15.predict(cals[tr], e, raw, nrm)
                res[LABELS[k], e] = m12.corp(p, y)["mcb"]
                pool[e][0].append(p); pool[e][1].append(y)
        for e in EST:
            res["pooled_oof", e] = m12.corp(np.concatenate(pool[e][0]), np.concatenate(pool[e][1]))["mcb"]
        return res

    full = {s: list(P.index[folds == s]) for s in ("dl19", "dl20")}
    cal_full = calibrators(full)
    point = mcb(cal_full, full)
    ref = pd.read_csv(OUT / "tab_calibration_diagnostics.csv")
    dev = max(abs(point[r.direction, {v: k for k, v in NAME.items()}[r.estimator]] - r.mcb) for r in ref.itertuples())
    print(f"full-sample MCB reproduces tab_calibration_diagnostics.csv: max |dev| = {dev:.2e}"); assert dev < 1e-12

    reps = {"fixed": [], "refit": []}
    for n, i in enumerate(boots):
        sel = {s: list(qids[i][folds[i] == s]) for s in ("dl19", "dl20")}
        reps["fixed"].append(mcb(cal_full, sel))
        reps["refit"].append(mcb(calibrators(sel), sel))
        if n % 500 == 0: print("replicate", n, flush=True)

    rows, crows = [], []
    for lab in LABELS:
        for e in EST:
            r = {"direction": lab, "estimator": NAME[e], "mcb": point[lab, e]}
            for m in ("fixed", "refit"):
                v = np.array([x[lab, e] for x in reps[m]]); lo, hi = np.percentile(v, [2.5, 97.5])
                r.update({f"{m}_median": np.median(v), f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_width": hi - lo})
            rows.append(r)
        for a, b in CON:
            r = {"direction": lab, "contrast": f"{NAME[a]} - {NAME[b]}", "delta_mcb": point[lab, a] - point[lab, b]}
            for m in ("fixed", "refit"):
                v = np.array([x[lab, a] - x[lab, b] for x in reps[m]]); lo, hi = np.percentile(v, [2.5, 97.5])
                r.update({f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_width": hi - lo, f"{m}_excl0": bool(lo > 0 or hi < 0)})
            r["significance_changed"] = r["fixed_excl0"] != r["refit_excl0"]
            crows.append(r)
    T, C = pd.DataFrame(rows), pd.DataFrame(crows)
    T.to_csv(OUT / "tab_refit_mcb.csv", index=False); C.to_csv(OUT / "tab_refit_mcb_contrasts.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False)); print(); print(C.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
