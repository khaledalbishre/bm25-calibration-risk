"""
Phase 2 (part 2) — calibrators, ECE/Brier, reliability diagram,
ranking-invariance check.

Fits `platt` and `isotonic` on DL19 judged pairs only; applies unchanged
to DL20. `norm` is a per-query min-max transform, computed directly on
whichever split it is applied to (it has no free parameters to "fit").

The pure-math helpers below (ece, brier, assert_ranking_invariant) are
self-tested in code/tests_synthetic.py against hand-built arrays with
known-correct answers.

Usage:
    python code/04_fit_calibrators.py
Reads:
    data/calib_frame_dl19.parquet, data/calib_frame_dl20.parquet
Writes:
    results/tab_calibration_dl20.csv
    results/fig_reliability_dl20.pdf / .png
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


# ---------------------------------------------------------------------
# Pure-math helpers (unit-tested in tests_synthetic.py)
# ---------------------------------------------------------------------

def norm_transform(df: pd.DataFrame) -> np.ndarray:
    """Per-query min-max normalization of bm25_score into [0, 1]."""
    out = np.empty(len(df))
    for qid, idx in df.groupby("qid").groups.items():
        scores = df.loc[idx, "bm25_score"].to_numpy()
        lo, hi = scores.min(), scores.max()
        if hi > lo:
            out[df.index.get_indexer(idx)] = (scores - lo) / (hi - lo)
        else:
            out[df.index.get_indexer(idx)] = 0.5  # degenerate: all tied
    return out


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> float:
    """Pooled, equal-width-bin Expected Calibration Error."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    n = len(probs)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost bin inclusive of 1.0
    bin_idx = np.clip(np.digitize(probs, bin_edges, right=False) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        total += (mask.sum() / n) * abs(acc - conf)
    return total


def per_query_ece(probs: np.ndarray, labels: np.ndarray, qids: np.ndarray, n_bins: int, min_judged: int = 5):
    """ECE computed within each query, then averaged across queries with
    >= min_judged judged docs. Returns (mean_ece, n_queries_used, n_excluded)."""
    df = pd.DataFrame({"p": probs, "y": labels, "qid": qids})
    per_q = []
    excluded = 0
    for qid, g in df.groupby("qid"):
        if len(g) < min_judged:
            excluded += 1
            continue
        per_q.append(ece(g["p"].to_numpy(), g["y"].to_numpy(), n_bins))
    return (float(np.mean(per_q)) if per_q else float("nan")), len(per_q), excluded


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probs - labels) ** 2))


def assert_ranking_invariant(df: pd.DataFrame, score_col: str, calibrated_col: str):
    """Within each query, the ordering induced by calibrated_col must be
    consistent with score_col (ties in calibrated allowed only where they
    don't reverse strict score order). Raises AssertionError on violation."""
    for qid, g in df.groupby("qid"):
        g = g.sort_values(score_col, ascending=False)
        cal = g[calibrated_col].to_numpy()
        # non-increasing calibrated score as raw score decreases
        diffs = np.diff(cal)
        if (diffs > 1e-9).any():
            raise AssertionError(f"ranking invariance violated for qid={qid}")


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

def main():
    train = pd.read_parquet(common.DATA_DIR / "calib_frame_dl19.parquet")
    test = pd.read_parquet(common.DATA_DIR / "calib_frame_dl20.parquet")

    # Fit on DL19
    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(train[["bm25_score"]].to_numpy(), train["label"].to_numpy())

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["bm25_score"].to_numpy(), train["label"].to_numpy())

    # Apply to DL20 (test)
    test = test.copy()
    test["p_norm"] = norm_transform(test)
    test["p_platt"] = platt.predict_proba(test[["bm25_score"]].to_numpy())[:, 1]
    test["p_isotonic"] = iso.predict(test["bm25_score"].to_numpy())

    rows = []
    for name, col in [("norm", "p_norm"), ("platt", "p_platt"), ("isotonic", "p_isotonic")]:
        e10 = ece(test[col].to_numpy(), test["label"].to_numpy(), 10)
        e15 = ece(test[col].to_numpy(), test["label"].to_numpy(), 15)
        pq_ece10, n_used, n_excl = per_query_ece(test[col].to_numpy(), test["label"].to_numpy(), test["qid"].to_numpy(), 10)
        b = brier(test[col].to_numpy(), test["label"].to_numpy())
        rows.append({
            "estimator": name, "ece_10bin_pooled": e10, "ece_15bin_pooled": e15,
            "ece_10bin_per_query_avg": pq_ece10, "per_query_n_used": n_used,
            "per_query_n_excluded_lt5judged": n_excl, "brier": b,
        })
        assert_ranking_invariant(test, "bm25_score", col)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(common.RESULTS_DIR / "tab_calibration_dl20.csv", index=False)
    print(out_df.to_string(index=False))
    print("Ranking invariance: PASSED for norm, platt, isotonic (no violations raised).")

    # Reliability diagram (3 curves)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
        for name, col in [("norm", "p_norm"), ("platt", "p_platt"), ("isotonic", "p_isotonic")]:
            bin_edges = np.linspace(0, 1, 11)
            bin_idx = np.clip(np.digitize(test[col], bin_edges, right=False) - 1, 0, 9)
            xs, ys = [], []
            for b in range(10):
                mask = bin_idx == b
                if mask.any():
                    xs.append(test.loc[mask, col].mean())
                    ys.append(test.loc[mask, "label"].mean())
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xlabel("mean predicted P(Rel)")
        ax.set_ylabel("observed relevance frequency")
        ax.set_title("Reliability diagram — DL20 (10 bins)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(common.RESULTS_DIR / "fig_reliability_dl20.pdf")
        fig.savefig(common.RESULTS_DIR / "fig_reliability_dl20.png", dpi=200)
        print("wrote results/fig_reliability_dl20.{pdf,png}")
    except Exception as e:
        print(f"reliability diagram plotting skipped: {e}")


if __name__ == "__main__":
    main()
