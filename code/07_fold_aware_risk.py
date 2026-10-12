"""
Fold-aware risk-detection analysis and depth-matched diagnostic.

Supersedes the pooled-AUROC comparison of 05_risk_signal.py /
06_diagnostics.py as the primary RQ3 analysis. Runs ONLY on existing
artefacts (TREC run files, calibration parquets, qrels): no index, no
retrieval, no new data.

Why a fold-aware protocol. In the two-fold out-of-fold design, DL19 is
scored by a calibrator fit on DL20 and DL20 by a calibrator fit on DL19.
Because the positive base rate among judged pairs differs between the
two collections (0.345 vs 0.174), the two calibrators output
systematically different probability levels (mean top-10 isotonic
P(Rel): 0.225 on DL19 vs 0.473 on DL20), while the risk label is
balanced within each fold (below-median nDCG@10). Pooling raw scores
from the two folds into one AUROC therefore mixes a between-fold level
offset into the ranking and penalises calibration-dependent signals.
The fold-aware estimate used here is the STRATIFIED AUROC: the
Mann-Whitney statistic computed only over (positive, negative) query
pairs from the same fold, i.e. the per-fold AUROCs averaged with
weights n_pos * n_neg. It is invariant to any fold-specific monotone
transformation of a signal. Confidence intervals come from a paired
bootstrap that resamples queries within each fold (stratified), 2000
resamples, fixed seed.

Signals (all oriented "higher = predicted better retrieval"; risk
score = -signal):
  level, raw score       top1, mean@10, mean@100
  dispersion, raw score  std@10, CV@10, std@100, CV@100
  calibrated, top-10     QRC-mean / QRC-max / std of P(Rel) for
                         platt, isotonic, norm+platt, norm+isotonic,
                         and the uncalibrated per-query min-max `norm`
CV@100 = std(top-100 scores) / mean(top-100 scores) is the NQC-style
score-dispersion baseline. It is a mean-normalised variant inspired by
NQC (Shtok et al.), NOT the canonical NQC, which normalises by the
retrieval score of the corpus treated as a single document.

Usage:
    python code/07_fold_aware_risk.py
Writes:
    results/tab_risk_foldaware.csv    per-signal AUROCs, CIs, deltas, Spearman
    results/tab_risk_contrasts.csv    paired depth-/aggregation-matched contrasts
    results/tab_fold_summary.csv      per-fold medians, prevalence, signal levels
    results/fig_risk_coverage_foldaware.pdf / .png
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")
m05 = import_module("05_risk_signal")

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

FOLDS = ("dl19", "dl20")
BASELINE = "cv100"


# ---------------------------------------------------------------------
# Pure-math helpers (unit-tested in tests_synthetic.py)
# ---------------------------------------------------------------------

def risk_auc(labels: np.ndarray, signal: np.ndarray) -> float:
    """Mann-Whitney AUROC for detecting labels==1 with risk = -signal.
    Ties receive half credit (identical to sklearn's roc_auc_score)."""
    labels = np.asarray(labels)
    r = rankdata(-np.asarray(signal, dtype=float))
    n1 = labels.sum()
    n0 = len(labels) - n1
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auc(labels: np.ndarray, signal: np.ndarray, folds: np.ndarray) -> float:
    """AUROC over within-fold (positive, negative) pairs only."""
    num = den = 0.0
    for f in np.unique(folds):
        m = folds == f
        n1 = labels[m].sum()
        n0 = m.sum() - n1
        if n1 == 0 or n0 == 0:
            continue
        num += risk_auc(labels[m], signal[m]) * n1 * n0
        den += n1 * n0
    return float(num / den)


def within_fold_rank(values: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Percentile rank of each value within its own fold."""
    out = np.empty(len(values))
    for f in np.unique(folds):
        m = folds == f
        out[m] = rankdata(values[m]) / m.sum()
    return out


def stratified_bootstrap_indices(folds: np.ndarray, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = [np.where(folds == f)[0] for f in np.unique(folds)]
    return [np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(n_resamples)]


# ---------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------

def load_run(split: str) -> pd.DataFrame:
    rows = []
    with open(common.RUNS_DIR / f"bm25_{split}.trec") as fh:
        for line in fh:
            qid, _, docid, rank, score, _tag = line.split()
            rows.append((qid, docid, int(rank), float(score)))
    df = pd.DataFrame(rows, columns=["qid", "docid", "rank", "bm25_score"])
    mm = df.groupby("qid")["bm25_score"].agg(["min", "max"])
    df = df.join(mm, on="qid")
    df["norm_score"] = ((df["bm25_score"] - df["min"]) / (df["max"] - df["min"])).clip(0, 1)
    return df.drop(columns=["min", "max"])


def norm_of_frame(frame: pd.DataFrame, run: pd.DataFrame) -> np.ndarray:
    """Min-max normalised score of judged pairs, using the full top-1000."""
    mm = run.groupby("qid")["bm25_score"].agg(["min", "max"])
    f = frame.join(mm, on="qid")
    return ((f["bm25_score"] - f["min"]) / (f["max"] - f["min"])).clip(0, 1).to_numpy()


def fold_signals(train_split: str, predict_split: str, runs: dict, frames: dict) -> pd.DataFrame:
    tr = frames[train_split]
    y = tr["label"].to_numpy()
    x = tr["bm25_score"].to_numpy()
    xn = norm_of_frame(tr, runs[train_split])

    cal = {
        "platt": LogisticRegression(penalty=None, max_iter=10000).fit(x.reshape(-1, 1), y),
        "iso": IsotonicRegression(out_of_bounds="clip").fit(x, y),
        "nplatt": LogisticRegression(penalty=None, max_iter=10000).fit(xn.reshape(-1, 1), y),
        "niso": IsotonicRegression(out_of_bounds="clip").fit(xn, y),
    }
    run = runs[predict_split].copy()
    run["p_platt"] = cal["platt"].predict_proba(run[["bm25_score"]].to_numpy())[:, 1]
    run["p_iso"] = cal["iso"].predict(run["bm25_score"].to_numpy())
    run["p_nplatt"] = cal["nplatt"].predict_proba(run[["norm_score"]].to_numpy())[:, 1]
    run["p_niso"] = cal["niso"].predict(run["norm_score"].to_numpy())
    run["p_norm"] = run["norm_score"]

    out = []
    for qid, g in run.groupby("qid"):
        g = g.sort_values("rank")
        t10 = g.head(common.QRC_K)
        t100 = g.head(common.NQC_K)
        s10, s100 = t10["bm25_score"].to_numpy(), t100["bm25_score"].to_numpy()
        row = {
            "qid": qid, "fold": predict_split,
            "top1": s10[0], "mean10": s10.mean(), "mean100": s100.mean(),
            "std10": s10.std(ddof=0), "cv10": m05.nqc(s10),
            "std100": s100.std(ddof=0), "cv100": m05.nqc(s100),
        }
        for est in ("norm", "platt", "iso", "nplatt", "niso"):
            p = t10[f"p_{est}"].to_numpy()
            row[f"qrcmean_{est}"] = p.mean()
            row[f"qrcmax_{est}"] = p.max()
            row[f"pstd10_{est}"] = p.std(ddof=0)
        out.append(row)
    return pd.DataFrame(out).set_index("qid")


# Display metadata: (column, label, family, depth)
SIGNALS = [
    ("cv100", "CV@100 (NQC-style baseline)", "dispersion", 100),
    ("std100", "std@100", "dispersion", 100),
    ("mean100", "mean@100", "level", 100),
    ("cv10", "CV@10", "dispersion", 10),
    ("std10", "std@10", "dispersion", 10),
    ("mean10", "mean@10", "level", 10),
    ("top1", "top-1 score", "level", 1),
    ("qrcmean_norm", "QRC-mean (norm, uncalibrated)", "calibrated level", 10),
    ("qrcmean_platt", "QRC-mean (Platt)", "calibrated level", 10),
    ("qrcmean_iso", "QRC-mean (isotonic)", "calibrated level", 10),
    ("qrcmean_nplatt", "QRC-mean (norm+Platt)", "calibrated level", 10),
    ("qrcmean_niso", "QRC-mean (norm+isotonic)", "calibrated level", 10),
    ("qrcmax_platt", "QRC-max (Platt)", "calibrated level", 10),
    ("qrcmax_iso", "QRC-max (isotonic)", "calibrated level", 10),
    ("qrcmax_norm", "QRC-max (norm)", "calibrated level", 10),
    ("qrcmax_nplatt", "QRC-max (norm+Platt)", "calibrated level", 10),
    ("qrcmax_niso", "QRC-max (norm+isotonic)", "calibrated level", 10),
    ("pstd10_platt", "std of P(Rel)@10 (Platt)", "calibrated dispersion", 10),
    ("pstd10_iso", "std of P(Rel)@10 (isotonic)", "calibrated dispersion", 10),
    ("pstd10_niso", "std of P(Rel)@10 (norm+isotonic)", "calibrated dispersion", 10),
]

CONTRASTS = [
    # same depth (k=10): dispersion vs level
    ("std10", "mean10", "depth-matched @10: dispersion vs level (raw)"),
    ("cv10", "mean10", "depth-matched @10: CV vs level (raw)"),
    ("std10", "top1", "dispersion@10 vs top-1 level"),
    ("pstd10_iso", "qrcmean_iso", "calibrated @10: dispersion vs level (isotonic)"),
    ("pstd10_platt", "qrcmean_platt", "calibrated @10: dispersion vs level (Platt)"),
    # same depth (k=100): dispersion vs level
    ("std100", "mean100", "depth-matched @100: dispersion vs level (raw)"),
    # same statistic, deeper list
    ("cv100", "cv10", "depth effect: CV@100 vs CV@10"),
    ("std100", "std10", "depth effect: std@100 vs std@10"),
    # calibration effect at fixed aggregation
    ("qrcmean_platt", "mean10", "calibration effect: QRC-mean (Platt) vs raw mean@10"),
    ("qrcmean_nplatt", "qrcmean_platt", "better-calibrated vs raw-score calibrator (QRC-mean)"),
    ("qrcmean_niso", "qrcmean_iso", "better-calibrated vs raw-score calibrator (QRC-mean, isotonic)"),
    # QRC-max estimator dependence
    ("qrcmax_iso", "qrcmax_platt", "QRC-max: isotonic (ties) vs Platt (strict)"),
]


def main():
    runs = {s: load_run(s) for s in FOLDS}
    frames = {s: pd.read_parquet(common.DATA_DIR / f"calib_frame_{s}.parquet") for s in FOLDS}
    ndcg10 = {s: m05.per_query_ndcg10(s) for s in FOLDS}

    parts = []
    for train_split, predict_split in (("dl19", "dl20"), ("dl20", "dl19")):
        df = fold_signals(train_split, predict_split, runs, frames)
        df["ndcg10"] = pd.Series(ndcg10[predict_split])
        df["risk"] = (df["ndcg10"] < df["ndcg10"].median()).astype(int)
        parts.append(df)
    P = pd.concat(parts)
    assert len(P) == 97, len(P)

    folds = P["fold"].to_numpy()
    y = P["risk"].to_numpy()
    nd_rank = within_fold_rank(P["ndcg10"].to_numpy(), folds)
    boots = stratified_bootstrap_indices(folds, common.BOOTSTRAP_RESAMPLES, common.BOOTSTRAP_SEED)

    sig = {c: P[c].to_numpy(dtype=float) for c, *_ in SIGNALS}
    boot_auc = {c: np.array([stratified_auc(y[i], sig[c][i], folds[i]) for i in boots]) for c in sig}

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return float(lo), float(hi)

    base_point = stratified_auc(y, sig[BASELINE], folds)
    rows = []
    for col, label, family, depth in SIGNALS:
        s = sig[col]
        point = stratified_auc(y, s, folds)
        lo, hi = ci(boot_auc[col])
        dlo, dhi = ci(boot_auc[col] - boot_auc[BASELINE])
        row = {"signal": col, "label": label, "family": family, "depth": depth}
        for f in FOLDS:
            m = folds == f
            row[f"auroc_{f}"] = risk_auc(y[m], s[m])
            row[f"spearman_{f}"] = spearmanr(s[m], P["ndcg10"].to_numpy()[m])[0] if np.ptp(s[m]) > 0 else np.nan
        rho, p = spearmanr(within_fold_rank(s, folds), nd_rank)
        row.update({
            "auroc_stratified": point, "auroc_ci_lo": lo, "auroc_ci_hi": hi,
            "delta_vs_cv100": point - base_point, "delta_ci_lo": dlo, "delta_ci_hi": dhi,
            "delta_ci_excludes_0": bool(dlo > 0 or dhi < 0),
            "spearman_pooled_withinfold": rho, "spearman_pooled_p": p,
            "auroc_pooled_unstratified_SECONDARY": risk_auc(y, s),
            "n_distinct_dl19": int(len(np.unique(np.round(s[folds == "dl19"], 12)))),
            "n_distinct_dl20": int(len(np.unique(np.round(s[folds == "dl20"], 12)))),
        })
        rows.append(row)
    tab = pd.DataFrame(rows)
    meta = {"n_dl19": int((folds == "dl19").sum()), "pos_dl19": int(y[folds == "dl19"].sum()),
            "n_dl20": int((folds == "dl20").sum()), "pos_dl20": int(y[folds == "dl20"].sum()),
            "n_bootstrap": len(boots)}
    for k, v in meta.items():
        tab[k] = v
    tab.to_csv(common.RESULTS_DIR / "tab_risk_foldaware.csv", index=False)

    crow = []
    for a, b, desc in CONTRASTS:
        d = stratified_auc(y, sig[a], folds) - stratified_auc(y, sig[b], folds)
        lo, hi = ci(boot_auc[a] - boot_auc[b])
        crow.append({"signal_a": a, "signal_b": b, "contrast": desc, "delta_auroc": d,
                     "ci_lo": lo, "ci_hi": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})
    ctab = pd.DataFrame(crow)
    ctab.to_csv(common.RESULTS_DIR / "tab_risk_contrasts.csv", index=False)

    pd.set_option("display.width", 220)
    print(tab[["label", "auroc_dl19", "auroc_dl20", "auroc_stratified", "auroc_ci_lo", "auroc_ci_hi",
               "delta_vs_cv100", "delta_ci_lo", "delta_ci_hi", "spearman_pooled_withinfold",
               "auroc_pooled_unstratified_SECONDARY"]].round(3).to_string(index=False))
    print()
    print(ctab.round(3).to_string(index=False))
    # Per-fold summary: risk-target medians, prevalence, and the mean level
    # of calibrated signals (documents the between-fold offset).
    summ = P.groupby("fold").agg(
        n_queries=("risk", "size"), n_at_risk=("risk", "sum"),
        median_ndcg10=("ndcg10", "median"), n_ndcg10_zero=("ndcg10", lambda v: int((v == 0).sum())),
        mean_qrcmean_iso=("qrcmean_iso", "mean"), mean_qrcmean_platt=("qrcmean_platt", "mean"),
        mean_qrcmean_niso=("qrcmean_niso", "mean"), mean_cv100=("cv100", "mean"))
    summ.to_csv(common.RESULTS_DIR / "tab_fold_summary.csv")

    print("\nfold means of selected signals (level offset between folds):")
    print(P.groupby("fold")[["qrcmean_iso", "qrcmean_platt", "qrcmean_niso", "cv100"]].mean().round(3).to_string())

    # Fold-aware risk-coverage curve: signals ranked within fold, then pooled.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        for col, name in [("cv100", "CV@100 (NQC-style)"), ("std10", "std@10"),
                          ("qrcmean_iso", "QRC-mean (isotonic)"), ("top1", "top-1 score")]:
            order = np.argsort(-within_fold_rank(sig[col], folds), kind="stable")
            bad = np.cumsum(y[order])
            n = np.arange(1, len(order) + 1)
            ax.plot(n / len(order), bad / n, label=name)
        ax.axhline(y.mean(), color="grey", linestyle=":", linewidth=1, label="no selection")
        ax.set_xlabel("coverage (fraction of queries retained, most confident first)")
        ax.set_ylabel("below-median nDCG@10 rate among retained")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(common.RESULTS_DIR / "fig_risk_coverage_foldaware.pdf")
        fig.savefig(common.RESULTS_DIR / "fig_risk_coverage_foldaware.png", dpi=200)
        print("wrote results/fig_risk_coverage_foldaware.{pdf,png}")
    except Exception as e:  # plotting is optional
        print(f"risk-coverage plotting skipped: {e}")


if __name__ == "__main__":
    main()
