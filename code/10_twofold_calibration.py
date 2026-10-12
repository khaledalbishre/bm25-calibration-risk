"""
Symmetric two-fold calibration evaluation (paper Table 3).

Mirrors the two-fold risk protocol of 07_fold_aware_risk.py: calibrators
are fitted on one collection's judged pairs and evaluated on the other's,
in both directions (DL19 -> DL20 and DL20 -> DL19).

Normalization (authoritative definition): per-query min-max of the BM25
score over the query's full retrieved top-1000 list, which is available at
deployment time. It is used by `norm` and as the input of norm+Platt and
norm+isotonic. Metrics are computed on judged pairs only. The earlier
judged-pairs-only normalization of `norm` is reported as a sensitivity row
("norm (judged-pairs normalization)") and is not used elsewhere.

Calibrators, metrics and bins are the same as in 04_fit_calibrators.py:
unregularized logistic regression (Platt), isotonic regression with
clipping, equal-width-bin ECE (10 and 15 bins), per-query ECE over queries
with at least 5 judged pairs, and Brier score.

Usage:
    python code/10_twofold_calibration.py
Reads:
    data/calib_frame_{dl19,dl20}.parquet, runs/bm25_{dl19,dl20}.trec
Writes:
    results/tab_calibration_twofold.csv
        one row per (direction, estimator): ece_10bin_pooled,
        ece_15bin_pooled, ece_10bin_per_query_avg, brier, mean_pred,
        base_rate, n_pairs
    results/tab_calibration_pooled_oof.csv
        one row per estimator: brier_pooled_oof = Brier score of the
        out-of-fold predictions of both directions over all judged pairs
        (equivalently, the pair-count-weighted mean of the two fold Brier
        scores)
    results/tab_calibration_brier_ci.csv
        Brier score with 95% query-clustered paired bootstrap interval per
        (direction or pooled_oof, estimator); see brier_bootstrap()
    results/tab_calibration_brier_contrasts.csv
        paired Brier differences between estimators with 95% intervals
"""
import sys
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
common = import_module("00_common")
m04 = import_module("04_fit_calibrators")
m07 = import_module("07_fold_aware_risk")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

DIRECTIONS = (("dl19", "dl20"), ("dl20", "dl19"))
ESTIMATORS = ["norm", "platt", "isotonic", "norm+platt", "norm+isotonic"]
SENSITIVITY = "norm (judged-pairs normalization)"


def predictions(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    y = train["label"].to_numpy()
    platt = LogisticRegression(penalty=None, max_iter=10000).fit(train[["bm25_score"]].to_numpy(), y)
    iso = IsotonicRegression(out_of_bounds="clip").fit(train["bm25_score"].to_numpy(), y)
    nplatt = LogisticRegression(penalty=None, max_iter=10000).fit(train[["norm_full"]].to_numpy(), y)
    niso = IsotonicRegression(out_of_bounds="clip").fit(train["norm_full"].to_numpy(), y)
    return {
        "norm": test["norm_full"].to_numpy(),
        "platt": platt.predict_proba(test[["bm25_score"]].to_numpy())[:, 1],
        "isotonic": iso.predict(test["bm25_score"].to_numpy()),
        "norm+platt": nplatt.predict_proba(test[["norm_full"]].to_numpy())[:, 1],
        "norm+isotonic": niso.predict(test["norm_full"].to_numpy()),
        SENSITIVITY: m04.norm_transform(test),
    }


CONTRASTS = [("norm+platt", "norm"), ("norm+isotonic", "norm"), ("platt", "norm"), ("isotonic", "norm"),
             ("norm+platt", "platt"), ("norm+isotonic", "isotonic"), ("norm+platt", "norm+isotonic")]


def brier_bootstrap(frames: dict, oof: dict):
    """Query-clustered, paired percentile bootstrap of the Brier score.

    Resampling unit: a query of the evaluation collection, drawn with
    replacement together with all of its judged pairs (judged pairs of one
    query are not independent). Queries are resampled within each
    evaluation collection (stratified), and the same resampled queries are
    used for every estimator (paired). The pooled out-of-fold Brier of a
    resample is the total squared error over both collections' resampled
    pairs divided by their number. 95% percentile intervals,
    BOOTSTRAP_RESAMPLES resamples, seed BOOTSTRAP_SEED.
    """
    ests = ESTIMATORS
    sse, cnt = {}, {}
    for k, (tr, te) in enumerate(DIRECTIONS):
        qid = frames[te]["qid"].to_numpy()
        uq, inv = np.unique(qid, return_inverse=True)
        cnt[k] = np.bincount(inv, minlength=len(uq)).astype(float)
        for est in ests:
            p, y = oof[est][k]
            sse[k, est] = np.bincount(inv, weights=(p - y) ** 2, minlength=len(uq))
    rng = np.random.default_rng(common.BOOTSTRAP_SEED)
    draws = [[rng.integers(0, len(cnt[k]), len(cnt[k])) for k in range(len(DIRECTIONS))]
             for _ in range(common.BOOTSTRAP_RESAMPLES)]
    labels = [f"{tr}->{te}" for tr, te in DIRECTIONS] + ["pooled_oof"]

    def stat(k, est, idx):
        if k < len(DIRECTIONS):
            return sse[k, est][idx[k]].sum() / cnt[k][idx[k]].sum()
        return (sum(sse[j, est][idx[j]].sum() for j in range(len(DIRECTIONS)))
                / sum(cnt[j][idx[j]].sum() for j in range(len(DIRECTIONS))))

    full = [np.arange(len(cnt[k])) for k in range(len(DIRECTIONS))]
    boot = {(k, e): np.array([stat(k, e, d) for d in draws]) for k in range(len(labels)) for e in ests}
    ci_rows, con_rows = [], []
    for k, lab in enumerate(labels):
        for e in ests:
            lo, hi = np.percentile(boot[k, e], [2.5, 97.5])
            ci_rows.append({"direction": lab, "estimator": e, "brier": stat(k, e, full),
                            "ci_lo": lo, "ci_hi": hi, "n_queries": int(sum(len(f) for f in full))
                            if k == len(DIRECTIONS) else len(full[k])})
        for a, b in CONTRASTS:
            d = boot[k, a] - boot[k, b]
            lo, hi = np.percentile(d, [2.5, 97.5])
            con_rows.append({"direction": lab, "contrast": f"{a} - {b}",
                             "delta_brier": stat(k, a, full) - stat(k, b, full), "ci_lo": lo, "ci_hi": hi,
                             "ci_excludes_0": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(ci_rows), pd.DataFrame(con_rows)


def main():
    runs = {s: m07.load_run(s) for s in ("dl19", "dl20")}
    frames = {}
    for s in ("dl19", "dl20"):
        f = pd.read_parquet(common.DATA_DIR / f"calib_frame_{s}.parquet").copy()
        f["norm_full"] = m07.norm_of_frame(f, runs[s])
        frames[s] = f

    rows, oof = [], {}
    for tr, te in DIRECTIONS:
        test = frames[te]
        y = test["label"].to_numpy()
        for est, p in predictions(frames[tr], test).items():
            m04.assert_ranking_invariant(test.assign(p=p), "bm25_score", "p")
            pq, n_used, _ = m04.per_query_ece(p, y, test["qid"].to_numpy(), 10)
            rows.append({"direction": f"{tr}->{te}", "estimator": est,
                         "ece_10bin_pooled": m04.ece(p, y, 10), "ece_15bin_pooled": m04.ece(p, y, 15),
                         "ece_10bin_per_query_avg": pq, "per_query_n_used": n_used,
                         "brier": m04.brier(p, y), "mean_pred": float(p.mean()),
                         "base_rate": float(y.mean()), "n_pairs": len(p)})
            oof.setdefault(est, []).append((p, y))
    tab = pd.DataFrame(rows)
    tab.to_csv(common.RESULTS_DIR / "tab_calibration_twofold.csv", index=False)

    pooled = []
    for est, parts in oof.items():
        p = np.concatenate([a for a, _ in parts])
        y = np.concatenate([b for _, b in parts])
        pooled.append({"estimator": est, "brier_pooled_oof": m04.brier(p, y), "n_pairs": len(p)})
    ptab = pd.DataFrame(pooled)
    ptab.to_csv(common.RESULTS_DIR / "tab_calibration_pooled_oof.csv", index=False)

    ci_tab, con_tab = brier_bootstrap(frames, oof)
    ci_tab.to_csv(common.RESULTS_DIR / "tab_calibration_brier_ci.csv", index=False)
    con_tab.to_csv(common.RESULTS_DIR / "tab_calibration_brier_contrasts.csv", index=False)
    print(ci_tab.round(4).to_string(index=False))
    print(con_tab.round(4).to_string(index=False))

    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    print(ptab.round(4).to_string(index=False))
    print("wrote results/tab_calibration_twofold.csv, results/tab_calibration_pooled_oof.csv")


if __name__ == "__main__":
    main()
