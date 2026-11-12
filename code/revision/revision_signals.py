"""
Query-level signals for the supplementary analyses: the ORIGINAL top-10 signals
of 07_fold_aware_risk.fold_signals() (unchanged) plus calibrated aggregates
over the top 100, computed with calibrators fitted exactly as in 07.
Also exposes the original bootstrap indices so results are paired with
the published Tables 5-7.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from revision_common import common, m05, m07, FOLDS

EST = ("norm", "platt", "iso", "nplatt", "niso")


def fit_calibrators(train_frame, train_run):
    y = train_frame["label"].to_numpy()
    x = train_frame["bm25_score"].to_numpy()
    xn = m07.norm_of_frame(train_frame, train_run)
    return {
        "platt": LogisticRegression(penalty=None, max_iter=10000).fit(x.reshape(-1, 1), y),
        "iso": IsotonicRegression(out_of_bounds="clip").fit(x, y),
        "nplatt": LogisticRegression(penalty=None, max_iter=10000).fit(xn.reshape(-1, 1), y),
        "niso": IsotonicRegression(out_of_bounds="clip").fit(xn, y),
    }


def apply_calibrators(cal, run):
    run = run.copy()
    run["p_platt"] = cal["platt"].predict_proba(run[["bm25_score"]].to_numpy())[:, 1]
    run["p_iso"] = cal["iso"].predict(run["bm25_score"].to_numpy())
    run["p_nplatt"] = cal["nplatt"].predict_proba(run[["norm_score"]].to_numpy())[:, 1]
    run["p_niso"] = cal["niso"].predict(run["norm_score"].to_numpy())
    run["p_norm"] = run["norm_score"]
    return run


def depth_signals(run_cal, k):
    out = []
    for qid, g in run_cal.groupby("qid"):
        t = g.sort_values("rank").head(k)
        row = {"qid": qid}
        for e in EST:
            p = t[f"p_{e}"].to_numpy()
            row[f"qrcmean{k}_{e}"] = p.mean()
            row[f"pstd{k}_{e}"] = p.std(ddof=0)
        out.append(row)
    return pd.DataFrame(out).set_index("qid")


def load_inputs():
    runs = {s: m07.load_run(s) for s in FOLDS}
    frames = {s: pd.read_parquet(common.DATA_DIR / f"calib_frame_{s}.parquet") for s in FOLDS}
    ndcg10 = {s: m05.per_query_ndcg10(s) for s in FOLDS}
    return runs, frames, ndcg10


def build_P(runs, frames, ndcg10, frames_override=None):
    """Same row order as 07.main(): fit dl19 -> score dl20, then fit dl20 -> score dl19.
    frames_override lets the refit bootstrap pass resampled training frames."""
    parts = []
    for tr, pr in (("dl19", "dl20"), ("dl20", "dl19")):
        fr = frames if frames_override is None else {**frames, tr: frames_override[tr]}
        df = m07.fold_signals(tr, pr, runs, fr)
        cal = fit_calibrators(fr[tr], runs[tr])
        run_pr = runs[pr][runs[pr]["qid"].isin(df.index)]
        extra = depth_signals(apply_calibrators(cal, run_pr), 100)
        df = df.join(extra)
        df["ndcg10"] = pd.Series(ndcg10[pr])
        df["risk"] = (df["ndcg10"] < df["ndcg10"].median()).astype(int)
        parts.append(df)
    P = pd.concat(parts)
    assert len(P) == 97
    return P


def original_boots(folds):
    return m07.stratified_bootstrap_indices(folds, common.BOOTSTRAP_RESAMPLES, common.BOOTSTRAP_SEED)
