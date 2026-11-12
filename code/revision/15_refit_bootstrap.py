"""
Sensitivity of the main results to refitting the calibrators inside the
bootstrap (manuscript Appendix A.1, Table A1).
The original fixed-calibrator analyses are untouched; this script computes
fixed and refit intervals under IDENTICAL query resamples (the original 07
indices), so the only difference is refitting.
Writes results/revision/tab_refit_brier.csv, tab_refit_brier_contrasts.csv,
       tab_refit_auroc.csv, tab_refit_auroc_contrasts.csv
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from revision_common import common, m07, OUT, ROOT
from revision_signals import load_inputs, build_P, original_boots

REFIT = ("platt", "iso", "nplatt", "niso")
BEST = ["constant", "norm", "platt", "iso", "nplatt", "niso"]
BCON = [("nplatt", "norm"), ("niso", "norm"), ("platt", "norm"), ("iso", "norm"),
        ("nplatt", "platt"), ("niso", "iso"), ("platt", "constant"), ("iso", "constant"),
        ("nplatt", "constant"), ("niso", "constant")]
ASIG = ["cv100", "top1", "mean10", "qrcmean_norm", "qrcmean_platt", "qrcmean_iso", "qrcmean_nplatt",
        "qrcmean_niso", "qrcmax_platt", "qrcmax_iso", "pstd100_platt", "qrcmean100_platt"]
ACON = [("qrcmean_platt", "mean10"), ("qrcmean_nplatt", "qrcmean_platt"), ("qrcmean_niso", "qrcmean_iso"),
        ("qrcmax_iso", "qrcmax_platt"), ("qrcmean_nplatt", "qrcmean_norm")] + \
       [(s, "cv100") for s in ASIG if s != "cv100"]


def fit(x, xn, y):
    return {"platt": LogisticRegression(penalty=None, max_iter=10000).fit(x.reshape(-1, 1), y),
            "iso": IsotonicRegression(out_of_bounds="clip").fit(x, y),
            "nplatt": LogisticRegression(penalty=None, max_iter=10000).fit(xn.reshape(-1, 1), y),
            "niso": IsotonicRegression(out_of_bounds="clip").fit(xn, y),
            "constant": float(y.mean())}


def predict(cal, e, raw, nrm):
    if e == "constant": return np.full(raw.shape, cal["constant"])
    if e == "norm": return nrm
    x = nrm if e in ("nplatt", "niso") else raw
    flat = x.reshape(-1, 1) if e in ("platt", "nplatt") else x.reshape(-1)
    p = cal[e].predict_proba(flat)[:, 1] if e in ("platt", "nplatt") else cal[e].predict(flat)
    return p.reshape(raw.shape)


def main():
    runs, frames, ndcg10 = load_inputs()
    P = build_P(runs, frames, ndcg10)
    folds, y = P["fold"].to_numpy(), P["risk"].to_numpy()
    qids = P.index.to_numpy()
    boots = original_boots(folds)

    # per collection: judged pairs grouped by query; top-10/top-100 score matrices
    C = {}
    for s in ("dl19", "dl20"):
        f = frames[s].copy(); f["norm_full"] = m07.norm_of_frame(f, runs[s])
        r = runs[s].sort_values(["qid", "rank"])
        rows = P.index[folds == s]
        top = {k: r.groupby("qid").head(k) for k in (10, 100)}
        C[s] = {"frame": f, "pairs": {q: np.where(f["qid"].to_numpy() == q)[0] for q in rows},
                "raw": {k: np.vstack([top[k][top[k].qid == q]["bm25_score"].to_numpy() for q in rows]) for k in (10, 100)},
                "nrm": {k: np.vstack([top[k][top[k].qid == q]["norm_score"].to_numpy() for q in rows]) for k in (10, 100)},
                "rows": list(rows)}
    other = {"dl19": "dl20", "dl20": "dl19"}

    def world(sel_q):
        """sel_q[s] = list of resampled qids of collection s (None = full sample, fixed fit).
        Returns calibrators fitted on each collection's (resampled) judged pairs."""
        cals = {}
        for s in ("dl19", "dl20"):
            f = C[s]["frame"]
            idx = np.concatenate([C[s]["pairs"][q] for q in sel_q[s]])
            cals[s] = fit(f["bm25_score"].to_numpy()[idx], f["norm_full"].to_numpy()[idx], f["label"].to_numpy()[idx])
        return cals

    def signals(cals):
        """Calibrator-dependent signals for all 97 queries, eval collection s scored by cals[other[s]]."""
        out = {}
        for s in ("dl19", "dl20"):
            cal = cals[other[s]]
            for e in REFIT:
                p10 = predict(cal, e, C[s]["raw"][10], C[s]["nrm"][10])
                out.setdefault(f"qrcmean_{e}", {})[s] = p10.mean(1)
                out.setdefault(f"qrcmax_{e}", {})[s] = p10.max(1)
            p100 = predict(cal, "platt", C[s]["raw"][100], C[s]["nrm"][100])
            out.setdefault("pstd100_platt", {})[s] = p100.std(1)
            out.setdefault("qrcmean100_platt", {})[s] = p100.mean(1)
        sig = {}
        for c, d in out.items():
            v = np.empty(len(P))
            for s in ("dl19", "dl20"):
                v[folds == s] = d[s]
            sig[c] = v
        return sig

    def brier_parts(cals, sel_eval):
        """{(direction, est): (sse, n)} over the (resampled) eval queries."""
        res = {}
        for tr, te in (("dl19", "dl20"), ("dl20", "dl19")):
            f = C[te]["frame"]; idx = np.concatenate([C[te]["pairs"][q] for q in sel_eval[te]])
            raw, nrm, lab = f["bm25_score"].to_numpy()[idx], f["norm_full"].to_numpy()[idx], f["label"].to_numpy()[idx]
            for e in BEST:
                p = predict(cals[tr], e, raw, nrm)
                res[f"{tr}->{te}", e] = (float(((p - lab) ** 2).sum()), len(lab))
        for e in BEST:
            a, b = res["dl19->dl20", e], res["dl20->dl19", e]
            res["pooled_oof", e] = (a[0] + b[0], a[1] + b[1])
        return {k: v[0] / v[1] for k, v in res.items()}

    full_sel = {s: C[s]["rows"] for s in ("dl19", "dl20")}
    cal_full = world(full_sel)
    sig_full = {c: P[c].to_numpy(float) for c in ASIG}
    chk = signals(cal_full)
    dev = max(np.max(np.abs(chk[c] - P[c].to_numpy(float))) for c in chk)
    print(f"full-sample refit reproduces original signals: max |dev| = {dev:.2e}"); assert dev < 1e-12
    br_full = brier_parts(cal_full, full_sel)
    ref = pd.read_csv(ROOT / "results" / "tab_calibration_twofold.csv")
    m = {"isotonic": "iso", "norm+platt": "nplatt", "norm+isotonic": "niso"}
    dev = max(abs(br_full[r.direction, m.get(r.estimator, r.estimator)] - r.brier)
              for r in ref.itertuples() if not r.estimator.startswith("norm (judged"))
    print(f"full-sample Brier reproduces Table 3: max |dev| = {dev:.2e}"); assert dev < 1e-12

    A = {"fixed": [], "refit": []}; Bb = {"fixed": [], "refit": []}
    for n, i in enumerate(boots):
        sel = {s: list(qids[i][folds[i] == s]) for s in ("dl19", "dl20")}
        yi, fi = y[i], folds[i]
        # fixed: original signals, original calibrators
        A["fixed"].append({c: m07.stratified_auc(yi, sig_full[c][i], fi) for c in ASIG})
        Bb["fixed"].append(brier_parts(cal_full, sel))
        # refit
        cal_b = world(sel)
        sb = {**sig_full, **signals(cal_b)}
        A["refit"].append({c: m07.stratified_auc(yi, sb[c][i], fi) for c in ASIG})
        Bb["refit"].append(brier_parts(cal_b, sel))
        if n % 250 == 0: print("replicate", n, flush=True)

    def summarize(full, reps, keys, cons, kind):
        rows, crows = [], []
        for k in keys:
            row = {"key": k if isinstance(k, str) else " ".join(k), "point": full[k]}
            for mode in ("fixed", "refit"):
                v = np.array([r[k] for r in reps[mode]])
                lo, hi = np.percentile(v, [2.5, 97.5])
                row.update({f"{mode}_median": np.median(v), f"{mode}_lo": lo, f"{mode}_hi": hi, f"{mode}_width": hi - lo})
            rows.append(row)
        for a, b in cons:
            row = {"contrast": f"{a} - {b}" if kind == "auc" else f"{a[0]}: {a[1]} - {b[1]}", "point": full[a] - full[b]}
            for mode in ("fixed", "refit"):
                v = np.array([r[a] - r[b] for r in reps[mode]])
                lo, hi = np.percentile(v, [2.5, 97.5])
                row.update({f"{mode}_lo": lo, f"{mode}_hi": hi, f"{mode}_width": hi - lo,
                            f"{mode}_excl0": bool(lo > 0 or hi < 0)})
            row["significance_changed"] = row["fixed_excl0"] != row["refit_excl0"]
            crows.append(row)
        return pd.DataFrame(rows), pd.DataFrame(crows)

    auc_full = {c: m07.stratified_auc(y, sig_full[c], folds) for c in ASIG}
    TA, CA = summarize(auc_full, A, ASIG, ACON, "auc")
    bkeys = [(d, e) for d in ("dl19->dl20", "dl20->dl19", "pooled_oof") for e in BEST]
    bcons = [((d, a), (d, b)) for d in ("dl19->dl20", "dl20->dl19", "pooled_oof") for a, b in BCON]
    TB, CB = summarize(br_full, Bb, bkeys, bcons, "brier")
    for df, name in ((TA, "tab_refit_auroc"), (CA, "tab_refit_auroc_contrasts"),
                     (TB, "tab_refit_brier"), (CB, "tab_refit_brier_contrasts")):
        df.to_csv(OUT / f"{name}.csv", index=False)
    pd.set_option("display.width", 250)
    for df in (TA, CA, TB, CB):
        print(df.round(4).to_string(index=False)); print()


if __name__ == "__main__":
    main()
