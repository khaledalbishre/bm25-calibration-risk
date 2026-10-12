"""
Bounded post-hoc diagnostic block.

Runs ONLY on the small artefacts already produced by the host run
(calibration parquets + TREC run files). No index, no retrieval, no new
data. Diagnostics:

  D1. Positive base rate (label=1 fraction) among judged pairs, DL19 vs
      DL20; and min/max/quantiles of Platt and isotonic predicted
      probabilities on DL20 (explains why 10- and 15-bin pooled ECE
      coincide for Platt).
  D2. Exactly two principled variants: `norm+platt` and `norm+isotonic`
      — the calibrator is fit on the QUERY-WISE MIN-MAX NORMALIZED score
      (normalization computed within each query's full top-1000 run
      list) instead of the raw score; still fit on DL19 only, applied
      unchanged to DL20. Rationale: raw BM25 scores are not comparable
      across queries, so a single global calibrator on the raw score is
      expected to underperform; normalizing first removes the
      cross-query scale problem while remaining monotone within query.
      This is a scoped diagnostic, not a new method. Reported in the
      same calibration table (appended rows) and in the RQ3 AUROC table
      (appended rows, same two-fold pooled protocol + bootstrap).
  D3. Programmatic sign-convention check for risk_auroc: verify that
      both QRC and NQC are oriented "higher signal = better retrieval"
      on the pooled 97 queries (Spearman rho vs per-query nDCG@10 > 0,
      and risk-AUROC(-signal) == 1 - quality-AUROC(+signal)).
  D4. (added 2026-08-17, second pass) "Dispersion after calibration":
      per-query standard deviation of calibrated P(Rel) over the top-10
      (std_p10), for isotonic and norm+isotonic, evaluated on the
      below-median target with the same pooled two-fold + bootstrap
      protocol and appended to the risk table. Purpose: isolate whether
      QRC's weakness comes from the mean/max AGGREGATION discarding
      dispersion, rather than from calibration itself. NOTE the sign
      convention differs from QRC/NQC here: like NQC, higher dispersion
      is hypothesized to indicate BETTER retrieval, so std_p10 is fed
      to risk_auroc with the same higher-is-better orientation.

Re-running this script is idempotent: previously appended rows are
replaced, not duplicated.

Usage:
    python code/06_diagnostics.py
Reads:
    data/calib_frame_dl19.parquet, data/calib_frame_dl20.parquet
    runs/bm25_dl19.trec, runs/bm25_dl20.trec, data/*qrels*.txt
Writes:
    results/tab_diagnostics.csv               (D1 numbers)
    results/tab_calibration_dl20.csv          (appends norm+platt, norm+isotonic rows)
    results/tab_risk_auroc.csv                (appends QRC rows for the two variants)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")
m04 = import_module("04_fit_calibrators")
m05 = import_module("05_risk_signal")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load_run_df(split: str) -> pd.DataFrame:
    rows = []
    with open(common.RUNS_DIR / f"bm25_{split}.trec") as fh:
        for line in fh:
            qid, _, docid, rank, score, tag = line.strip().split()
            rows.append({"qid": qid, "docid": docid, "rank": int(rank), "bm25_score": float(score)})
    return pd.DataFrame(rows)


def run_minmax(run_df: pd.DataFrame) -> dict:
    """{qid: (min, max)} of bm25_score over the query's full top-1000."""
    g = run_df.groupby("qid")["bm25_score"]
    return {qid: (lo, hi) for qid, lo, hi in zip(g.min().index, g.min().values, g.max().values)}


def add_norm_col(df: pd.DataFrame, minmax: dict, col_out: str = "norm_score") -> pd.DataFrame:
    df = df.copy()
    lo = df["qid"].map(lambda q: minmax[q][0])
    hi = df["qid"].map(lambda q: minmax[q][1])
    rng = (hi - lo).replace(0, np.nan)
    df[col_out] = ((df["bm25_score"] - lo) / rng).fillna(0.5).clip(0.0, 1.0)
    return df


def quantile_report(name, p):
    qs = np.quantile(p, [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    labels = ["min", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "max"]
    return {f"{name}_{l}": float(v) for l, v in zip(labels, qs)}


def main():
    train = pd.read_parquet(common.DATA_DIR / "calib_frame_dl19.parquet")
    test = pd.read_parquet(common.DATA_DIR / "calib_frame_dl20.parquet")
    run19, run20 = load_run_df("dl19"), load_run_df("dl20")
    mm19, mm20 = run_minmax(run19), run_minmax(run20)

    diag_rows = {}

    # ---------------- D1: base rates + predicted-probability ranges ----
    br19 = float(train["label"].mean())
    br20 = float(test["label"].mean())
    diag_rows["base_rate_dl19"] = br19
    diag_rows["base_rate_dl20"] = br20
    print(f"D1 base rate (label=1 fraction among judged pairs): DL19={br19:.4f} ({train['label'].sum()}/{len(train)}), DL20={br20:.4f} ({test['label'].sum()}/{len(test)})")

    platt = LogisticRegression(penalty=None, max_iter=10000)
    platt.fit(train[["bm25_score"]].to_numpy(), train["label"].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(train["bm25_score"].to_numpy(), train["label"].to_numpy())

    p_platt = platt.predict_proba(test[["bm25_score"]].to_numpy())[:, 1]
    p_iso = iso.predict(test["bm25_score"].to_numpy())
    # Calibration-in-the-large: mean prediction vs base rate on DL20.
    diag_rows["platt_dl20_mean_pred"] = float(p_platt.mean())
    diag_rows["isotonic_dl20_mean_pred"] = float(p_iso.mean())
    diag_rows.update(quantile_report("platt_dl20", p_platt))
    diag_rows.update(quantile_report("isotonic_dl20", p_iso))
    print("D1 Platt predicted-prob quantiles on DL20:", {k.split('_')[-1]: round(v, 4) for k, v in quantile_report('platt_dl20', p_platt).items()})
    print("D1 isotonic predicted-prob quantiles on DL20:", {k.split('_')[-1]: round(v, 4) for k, v in quantile_report('isotonic_dl20', p_iso).items()})

    # Cross-check: this refit reproduces the Platt Brier score of
    # 04_fit_calibrators.py (column names keep their historical labels).
    host_tab = pd.read_csv(common.RESULTS_DIR / "tab_calibration_dl20.csv")
    host_brier_platt = float(host_tab.loc[host_tab["estimator"] == "platt", "brier"].iloc[0])
    sand_brier_platt = m04.brier(p_platt, test["label"].to_numpy())
    print(f"D1 cross-check: host platt Brier={host_brier_platt:.6f}, sandbox refit={sand_brier_platt:.6f}, |diff|={abs(host_brier_platt - sand_brier_platt):.2e}")
    diag_rows["platt_brier_host_vs_sandbox_absdiff"] = abs(host_brier_platt - sand_brier_platt)

    # Which ECE bins are populated for platt (why 10-bin == 15-bin):
    for nb in (10, 15):
        edges = np.linspace(0, 1, nb + 1)
        occupied = np.unique(np.clip(np.digitize(p_platt, edges) - 1, 0, nb - 1))
        print(f"D1 platt occupied {nb}-bins: {occupied.tolist()} (bin width {1/nb:.4f})")

    # ---------------- D2: norm+platt / norm+isotonic --------------------
    train_n = add_norm_col(train, mm19)
    test_n = add_norm_col(test, mm20)

    np_platt = LogisticRegression(penalty=None, max_iter=10000)
    np_platt.fit(train_n[["norm_score"]].to_numpy(), train_n["label"].to_numpy())
    np_iso = IsotonicRegression(out_of_bounds="clip")
    np_iso.fit(train_n["norm_score"].to_numpy(), train_n["label"].to_numpy())

    test_n = test_n.copy()
    test_n["p_normplatt"] = np_platt.predict_proba(test_n[["norm_score"]].to_numpy())[:, 1]
    test_n["p_normiso"] = np_iso.predict(test_n["norm_score"].to_numpy())

    new_cal_rows = []
    for name, col in [("norm+platt", "p_normplatt"), ("norm+isotonic", "p_normiso")]:
        e10 = m04.ece(test_n[col].to_numpy(), test_n["label"].to_numpy(), 10)
        e15 = m04.ece(test_n[col].to_numpy(), test_n["label"].to_numpy(), 15)
        pq10, n_used, n_excl = m04.per_query_ece(test_n[col].to_numpy(), test_n["label"].to_numpy(), test_n["qid"].to_numpy(), 10)
        b = m04.brier(test_n[col].to_numpy(), test_n["label"].to_numpy())
        m04.assert_ranking_invariant(test_n, "bm25_score", col)  # monotone in norm_score, norm_score monotone in raw within query
        new_cal_rows.append({
            "estimator": name, "ece_10bin_pooled": e10, "ece_15bin_pooled": e15,
            "ece_10bin_per_query_avg": pq10, "per_query_n_used": n_used,
            "per_query_n_excluded_lt5judged": n_excl, "brier": b,
        })
        print(f"D2 {name}: ECE10={e10:.4f} ECE15={e15:.4f} perq-ECE10={pq10:.4f} Brier={b:.4f} (ranking invariance PASSED)")

    base_cal = host_tab[~host_tab["estimator"].isin([r["estimator"] for r in new_cal_rows])]
    cal_tab = pd.concat([base_cal, pd.DataFrame(new_cal_rows)], ignore_index=True)
    cal_tab.to_csv(common.RESULTS_DIR / "tab_calibration_dl20.csv", index=False)

    # RQ3 re-run for the two variants: two-fold swap pooled out-of-fold.
    ndcg10 = {s: m05.per_query_ndcg10(s) for s in ("dl19", "dl20")}
    runs = {"dl19": run19, "dl20": run20}
    mms = {"dl19": mm19, "dl20": mm20}
    frames = {"dl19": train, "dl20": test}

    def fold_signals(train_split, predict_split, variant):
        tr = add_norm_col(frames[train_split], mms[train_split])
        if variant == "norm+platt":
            cal = LogisticRegression(penalty=None, max_iter=10000)
            cal.fit(tr[["norm_score"]].to_numpy(), tr["label"].to_numpy())
            predict = lambda x: cal.predict_proba(x.reshape(-1, 1))[:, 1]
        else:
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(tr["norm_score"].to_numpy(), tr["label"].to_numpy())
            predict = lambda x: cal.predict(x)
        run_df = add_norm_col(runs[predict_split], mms[predict_split])
        run_df["p"] = predict(run_df["norm_score"].to_numpy())
        out = []
        for qid, g in run_df.groupby("qid"):
            g = g.sort_values("rank")
            top10 = g[g["rank"] <= common.QRC_K]
            top100 = g[g["rank"] <= common.NQC_K]
            out.append({"qid": qid,
                        "qrc_mean": m05.qrc(top10["p"].to_numpy(), "mean"),
                        "qrc_max": m05.qrc(top10["p"].to_numpy(), "max"),
                        "nqc": m05.nqc(top100["bm25_score"].to_numpy())})
        df = pd.DataFrame(out).set_index("qid")
        nd = pd.Series(ndcg10[predict_split], name="ndcg10")
        df = df.join(nd, how="inner")
        median = df["ndcg10"].median()
        df["risk_primary"] = (df["ndcg10"] < median).astype(int)
        df["risk_secondary"] = (df["ndcg10"] == 0).astype(int)
        return df

    risk_tab = pd.read_csv(common.RESULTS_DIR / "tab_risk_auroc.csv")
    new_risk_rows = []
    pooled_by_variant = {}
    for variant in ("norm+platt", "norm+isotonic"):
        pooled = pd.concat([fold_signals("dl19", "dl20", variant), fold_signals("dl20", "dl19", variant)])
        assert len(pooled) == 97, f"expected 97 pooled queries, got {len(pooled)}"
        pooled_by_variant[variant] = pooled
        for target_col, target_name in [("risk_primary", "below_median"), ("risk_secondary", "ndcg10_eq_0")]:
            labels = pooled[target_col].to_numpy()
            if labels.min() == labels.max():
                continue
            for qrc_col, agg in [("qrc_mean", "mean"), ("qrc_max", "max")]:
                res = m05.bootstrap_ci(pooled[qrc_col].to_numpy(), pooled["nqc"].to_numpy(), labels,
                                        common.BOOTSTRAP_RESAMPLES, common.BOOTSTRAP_SEED)
                row = {
                    "risk_target": target_name, "estimator": f"QRC_{agg}[{variant}]",
                    "auroc": res["auroc_a"], "auroc_ci_lo": res["auroc_a_ci"][0], "auroc_ci_hi": res["auroc_a_ci"][1],
                    "auroc_nqc": res["auroc_b"], "auroc_nqc_ci_lo": res["auroc_b_ci"][0], "auroc_nqc_ci_hi": res["auroc_b_ci"][1],
                    "delta_auroc": res["delta"], "delta_ci_lo": res["delta_ci"][0], "delta_ci_hi": res["delta_ci"][1],
                    "n_queries": len(labels), "n_valid_bootstrap": res["n_valid_resamples"],
                }
                new_risk_rows.append(row)
                print(f"D2 RQ3 {target_name} QRC_{agg}[{variant}]: AUROC={res['auroc_a']:.4f} [{res['auroc_a_ci'][0]:.4f},{res['auroc_a_ci'][1]:.4f}] "
                      f"dAUROC={res['delta']:.4f} [{res['delta_ci'][0]:.4f},{res['delta_ci'][1]:.4f}]")

    # ---------------- D4: dispersion after calibration ------------------
    # std_p10 = per-query std (ddof=0) of calibrated P(Rel) over top-10.
    # Two variants: isotonic on the raw score, isotonic on the normalized
    # score. Same two-fold pooled protocol; below-median target only
    # (the stress target is underpowered for every signal).
    def fold_std_signals(train_split, predict_split, variant):
        tr_frame = frames[train_split]
        run_df = runs[predict_split].copy()
        if variant == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(tr_frame["bm25_score"].to_numpy(), tr_frame["label"].to_numpy())
            run_df["p"] = cal.predict(run_df["bm25_score"].to_numpy())
        else:  # norm+isotonic
            tr = add_norm_col(tr_frame, mms[train_split])
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(tr["norm_score"].to_numpy(), tr["label"].to_numpy())
            run_df = add_norm_col(run_df, mms[predict_split])
            run_df["p"] = cal.predict(run_df["norm_score"].to_numpy())
        out = []
        for qid, g in run_df.groupby("qid"):
            top10 = g.sort_values("rank").head(common.QRC_K)
            out.append({"qid": qid,
                        "std_p10": float(np.std(top10["p"].to_numpy(), ddof=0)),
                        "nqc": m05.nqc(g.sort_values("rank").head(common.NQC_K)["bm25_score"].to_numpy())})
        df = pd.DataFrame(out).set_index("qid")
        nd = pd.Series(ndcg10[predict_split], name="ndcg10")
        df = df.join(nd, how="inner")
        median = df["ndcg10"].median()
        df["risk_primary"] = (df["ndcg10"] < median).astype(int)
        return df

    for variant in ("isotonic", "norm+isotonic"):
        pooled = pd.concat([fold_std_signals("dl19", "dl20", variant),
                            fold_std_signals("dl20", "dl19", variant)])
        assert len(pooled) == 97
        labels = pooled["risk_primary"].to_numpy()
        rho, pval = spearmanr(pooled["std_p10"], pooled["ndcg10"])
        res = m05.bootstrap_ci(pooled["std_p10"].to_numpy(), pooled["nqc"].to_numpy(), labels,
                                common.BOOTSTRAP_RESAMPLES, common.BOOTSTRAP_SEED)
        new_risk_rows.append({
            "risk_target": "below_median", "estimator": f"std_p10[{variant}]",
            "auroc": res["auroc_a"], "auroc_ci_lo": res["auroc_a_ci"][0], "auroc_ci_hi": res["auroc_a_ci"][1],
            "auroc_nqc": res["auroc_b"], "auroc_nqc_ci_lo": res["auroc_b_ci"][0], "auroc_nqc_ci_hi": res["auroc_b_ci"][1],
            "delta_auroc": res["delta"], "delta_ci_lo": res["delta_ci"][0], "delta_ci_hi": res["delta_ci"][1],
            "n_queries": len(labels), "n_valid_bootstrap": res["n_valid_resamples"],
        })
        print(f"D4 below_median std_p10[{variant}]: AUROC={res['auroc_a']:.4f} [{res['auroc_a_ci'][0]:.4f},{res['auroc_a_ci'][1]:.4f}] "
              f"dAUROC vs NQC={res['delta']:.4f} [{res['delta_ci'][0]:.4f},{res['delta_ci'][1]:.4f}]  "
              f"Spearman rho vs nDCG@10={rho:.4f} (p={pval:.4g})")

    base_risk = risk_tab[~risk_tab["estimator"].isin([r["estimator"] for r in new_risk_rows])]
    risk_tab = pd.concat([base_risk, pd.DataFrame(new_risk_rows)], ignore_index=True)
    risk_tab.to_csv(common.RESULTS_DIR / "tab_risk_auroc.csv", index=False)

    # ---------------- D3: sign-convention check ------------------------
    # Rebuild the ORIGINAL (raw-score isotonic) pooled signals exactly as
    # 05_risk_signal.py did, and check orientation.
    pooled_orig = pd.concat([m05.compute_split_signals("dl19", "dl20").join(pd.Series(ndcg10["dl20"], name="ndcg10"), how="inner"),
                             m05.compute_split_signals("dl20", "dl19").join(pd.Series(ndcg10["dl19"], name="ndcg10"), how="inner")])
    assert len(pooled_orig) == 97
    for sig in ("qrc_mean", "qrc_max", "nqc"):
        rho, pval = spearmanr(pooled_orig[sig], pooled_orig["ndcg10"])
        print(f"D3 Spearman rho({sig}, nDCG@10) = {rho:.4f} (p={pval:.4g}) -> {'higher signal = better retrieval CONFIRMED' if rho > 0 else 'ORIENTATION PROBLEM'}")
        diag_rows[f"spearman_{sig}_ndcg10"] = float(rho)
        diag_rows[f"spearman_{sig}_pval"] = float(pval)
        # risk-AUROC(-signal, below-median) must equal 1 - AUROC(+signal, below-median)
        med = np.median(pooled_orig["ndcg10"])
        below = (pooled_orig["ndcg10"] < med).astype(int).to_numpy()
        a_risk = m05.risk_auroc(below, pooled_orig[sig].to_numpy())
        a_plus = roc_auc_score(below, pooled_orig[sig].to_numpy())
        assert abs(a_risk - (1 - a_plus)) < 1e-12, "risk_auroc sign convention violated"
        print(f"D3 {sig}: risk_auroc(-signal)={a_risk:.4f} == 1 - roc_auc(+signal)={1-a_plus:.4f}  OK")

    pd.DataFrame([diag_rows]).T.rename(columns={0: "value"}).to_csv(common.RESULTS_DIR / "tab_diagnostics.csv")
    print("wrote results/tab_diagnostics.csv; appended rows to tab_calibration_dl20.csv and tab_risk_auroc.csv")


if __name__ == "__main__":
    main()
