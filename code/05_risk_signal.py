"""
Phase 3 — QRC vs NQC risk-detection signal, pooled two-fold AUROC,
bootstrap CIs, risk-coverage curve.

Pure math helpers (nqc, qrc, auroc sign convention, bootstrap CI) are
self-tested in code/tests_synthetic.py.

Dispersion baseline formula: implemented as
    NQC(q) = std(top-k scores) / mean(top-k scores)
i.e. the coefficient-of-variation form of Shtok et al.'s normalized
query commitment, using the mean of the top-k retrieval scores as the
corpus-score proxy rather than a separate whole-collection score. This
is one of the two variants used in the QPP literature and is the one
computable purely from the BM25 run file (no extra collection-statistics
call). The manuscript labels it CV@100 rather than canonical NQC.

Risk score sign convention: both QRC and NQC are "confidence/coherence"
signals where higher = predicted more reliable. AUROC for risk DETECTION
is therefore computed with risk_score = -QRC (resp. -NQC) against the
binary "is this query low-quality" label.

"Median of the evaluated set" (primary risk target) is computed
per-direction: when predicting DL20 nDCG@10 from a DL19-fit calibrator,
the median is the median nDCG@10 *within DL20's 54 queries*; symmetric
for the DL19 direction. This keeps "the evaluated set" tied to whichever
split is actually being predicted on, which is the more natural reading
of "nDCG@10 below the median of the evaluated set." Recorded here as the
conservative choice per Phase 3 instructions; logged in decisions.md
when this actually runs.

Usage:
    python code/05_risk_signal.py
Reads:
    data/calib_frame_dl19.parquet, data/calib_frame_dl20.parquet
    results/tab_retrieval_eval.csv (per-query nDCG@10 needed -> this
        script recomputes per-query nDCG@10 via pytrec_eval directly,
        see per_query_ndcg10())
Writes:
    results/tab_risk_auroc.csv
    results/fig_risk_coverage.pdf / .png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------
# Pure-math helpers (unit-tested)
# ---------------------------------------------------------------------

def nqc(scores_topk: np.ndarray) -> float:
    """Coefficient-of-variation NQC; see module docstring for the
    exact-formula decision. Returns 0.0 for a degenerate all-equal top-k
    (std=0) rather than dividing by a possibly-zero mean unsafely."""
    scores_topk = np.asarray(scores_topk, dtype=float)
    mean = scores_topk.mean()
    if mean == 0:
        return 0.0
    return float(scores_topk.std(ddof=0) / mean)


def qrc(probs_topk: np.ndarray, agg: str = "mean") -> float:
    probs_topk = np.asarray(probs_topk, dtype=float)
    if agg == "mean":
        return float(probs_topk.mean())
    elif agg == "max":
        return float(probs_topk.max())
    raise ValueError(agg)


def risk_auroc(risk_labels: np.ndarray, signal: np.ndarray) -> float:
    """AUROC for detecting risk_labels==1 (low quality) using -signal
    (higher signal = higher predicted quality = lower predicted risk)."""
    return float(roc_auc_score(risk_labels, -np.asarray(signal, dtype=float)))


def bootstrap_ci(values_a: np.ndarray, values_b: np.ndarray, labels: np.ndarray,
                  n_resamples: int, seed: int):
    """Bootstrap over queries (paired). Returns dict with AUROC_a, AUROC_b,
    delta=AUROC_a-AUROC_b point estimates and 95% percentile CIs for each."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    auroc_a = risk_auroc(labels, values_a)
    auroc_b = risk_auroc(labels, values_b)
    delta = auroc_a - auroc_b

    boot_a, boot_b, boot_delta = [], [], []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        lab_s = labels[idx]
        if lab_s.min() == lab_s.max():
            continue  # AUROC undefined for a single-class resample; skip
        a = risk_auroc(lab_s, values_a[idx])
        b = risk_auroc(lab_s, values_b[idx])
        boot_a.append(a)
        boot_b.append(b)
        boot_delta.append(a - b)

    def ci(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return float(lo), float(hi)

    return {
        "auroc_a": auroc_a, "auroc_a_ci": ci(boot_a),
        "auroc_b": auroc_b, "auroc_b_ci": ci(boot_b),
        "delta": delta, "delta_ci": ci(boot_delta),
        "n_valid_resamples": len(boot_delta),
    }


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def per_query_ndcg10(split: str) -> dict:
    import pytrec_eval

    qrels = common.load_qrels(split)  # local TREC qrels files

    run = {}
    with open(common.RUNS_DIR / f"bm25_{split}.trec") as fh:
        for line in fh:
            qid, _, docid, rank, score, tag = line.strip().split()
            run.setdefault(qid, {})[docid] = float(score)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    per_query = evaluator.evaluate(run)
    return {qid: v["ndcg_cut_10"] for qid, v in per_query.items()}


def compute_split_signals(train_split: str, predict_split: str):
    """Fit platt+isotonic on train_split, apply to predict_split; compute
    QRC (mean, max over top-10) and NQC (over top-100) per query of
    predict_split. Returns a DataFrame indexed by qid."""
    train = pd.read_parquet(common.DATA_DIR / f"calib_frame_{train_split}.parquet")
    test = pd.read_parquet(common.DATA_DIR / f"calib_frame_{predict_split}.parquet")

    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(train[["bm25_score"]].to_numpy(), train["label"].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["bm25_score"].to_numpy(), train["label"].to_numpy())

    # QRC needs calibrated P(Rel) over the *full* top-10 of the run (not
    # just judged pairs), since unjudged docs still contribute to the
    # mean/max aggregate the searcher actually returned. Re-read the run
    # file directly rather than the judged-only calibration frame.
    run_rows = []
    with open(common.RUNS_DIR / f"bm25_{predict_split}.trec") as fh:
        for line in fh:
            qid, _, docid, rank, score, tag = line.strip().split()
            run_rows.append({"qid": qid, "rank": int(rank), "bm25_score": float(score)})
    run_df = pd.DataFrame(run_rows)
    run_df["p_isotonic"] = iso.predict(run_df["bm25_score"].to_numpy())

    out = []
    for qid, g in run_df.groupby("qid"):
        g_sorted = g.sort_values("rank")
        top10 = g_sorted[g_sorted["rank"] <= common.QRC_K]
        top100 = g_sorted[g_sorted["rank"] <= common.NQC_K]
        out.append({
            "qid": qid,
            "qrc_mean": qrc(top10["p_isotonic"].to_numpy(), "mean"),
            "qrc_max": qrc(top10["p_isotonic"].to_numpy(), "max"),
            "nqc": nqc(top100["bm25_score"].to_numpy()),
        })
    return pd.DataFrame(out).set_index("qid")


def main():
    ndcg10 = {s: per_query_ndcg10(s) for s in ("dl19", "dl20")}

    directions = [("dl19", "dl20"), ("dl20", "dl19")]  # (train, predict)
    pooled = []
    for train_split, predict_split in directions:
        sig = compute_split_signals(train_split, predict_split)
        nd = pd.Series(ndcg10[predict_split], name="ndcg10")
        df = sig.join(nd, how="inner")
        median = df["ndcg10"].median()
        df["risk_primary"] = (df["ndcg10"] < median).astype(int)
        df["risk_secondary"] = (df["ndcg10"] == 0).astype(int)
        df["predict_split"] = predict_split
        df["direction"] = f"{train_split}->{predict_split}"
        pooled.append(df)

    pooled_df = pd.concat(pooled)
    assert len(pooled_df) == 97, f"expected 97 pooled queries (43+54), got {len(pooled_df)}"

    rows = []
    for target_col, target_name in [("risk_primary", "below_median"), ("risk_secondary", "ndcg10_eq_0")]:
        labels = pooled_df[target_col].to_numpy()
        if labels.min() == labels.max():
            print(f"skip {target_name}: single-class labels (all {labels[0]}) in pooled set")
            continue
        for qrc_col, qrc_name in [("qrc_mean", "QRC_mean"), ("qrc_max", "QRC_max")]:
            res = bootstrap_ci(
                pooled_df[qrc_col].to_numpy(), pooled_df["nqc"].to_numpy(), labels,
                common.BOOTSTRAP_RESAMPLES, common.BOOTSTRAP_SEED,
            )
            rows.append({
                "risk_target": target_name, "estimator": qrc_name,
                "auroc": res["auroc_a"], "auroc_ci_lo": res["auroc_a_ci"][0], "auroc_ci_hi": res["auroc_a_ci"][1],
                "auroc_nqc": res["auroc_b"], "auroc_nqc_ci_lo": res["auroc_b_ci"][0], "auroc_nqc_ci_hi": res["auroc_b_ci"][1],
                "delta_auroc": res["delta"], "delta_ci_lo": res["delta_ci"][0], "delta_ci_hi": res["delta_ci"][1],
                "n_queries": len(labels), "n_valid_bootstrap": res["n_valid_resamples"],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(common.RESULTS_DIR / "tab_risk_auroc.csv", index=False)
    print(out_df.to_string(index=False))

    # CHECKPOINT 3
    bad = out_df[(out_df["auroc_nqc"] < 0.4) | (out_df["auroc_nqc"] > 0.98)]
    if len(bad):
        print("CHECKPOINT 3 WARNING: NQC AUROC outside plausible [0.4, 0.98] range — suspect label/sign bug.")
        print(bad.to_string(index=False))
    else:
        print("CHECKPOINT 3 PASSED: NQC AUROC values are plausible.")

    # Risk-coverage curve (primary target only, QRC_mean vs NQC)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pooled_df.copy()
        fig, ax = plt.subplots(figsize=(5, 5))
        for col, name in [("qrc_mean", "QRC (mean)"), ("nqc", "NQC")]:
            order = df.sort_values(col, ascending=False)  # most confident first
            cum_bad = np.cumsum(order["risk_primary"].to_numpy())
            coverage = np.arange(1, len(order) + 1) / len(order)
            risk_rate = cum_bad / np.arange(1, len(order) + 1)
            ax.plot(coverage, risk_rate, label=name)
        ax.set_xlabel("coverage (fraction of queries retained, most-confident-first)")
        ax.set_ylabel("cumulative fraction below-median nDCG@10")
        ax.set_title("Risk-coverage curve (primary target)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(common.RESULTS_DIR / "fig_risk_coverage.pdf")
        fig.savefig(common.RESULTS_DIR / "fig_risk_coverage.png", dpi=200)
        print("wrote results/fig_risk_coverage.{pdf,png}")
    except Exception as e:
        print(f"risk-coverage plotting skipped: {e}")


if __name__ == "__main__":
    main()
