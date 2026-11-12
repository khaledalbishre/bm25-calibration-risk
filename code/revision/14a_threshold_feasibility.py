"""
Absolute nDCG@10 thresholds, step 1: nDCG@10 distribution per fold and positive/negative counts for
candidate absolute thresholds. NO risk signal and NO AUROC is computed here.
Rule (fixed before any AUROC was computed): feasible iff >= 10 positives and >= 10 negatives
in each fold.
Writes results/revision/tab_ndcg10_distribution.csv, tab_threshold_counts.csv
"""
import numpy as np
import pandas as pd
from revision_common import common, m05, m07, OUT, FOLDS

MIN_PER_CLASS = 10
GRID = np.round(np.arange(0.05, 0.951, 0.05), 2)


def main():
    nd = {s: pd.Series(m05.per_query_ndcg10(s)) for s in FOLDS}
    dist = []
    for s in FOLDS:
        v = nd[s]
        dist.append({"fold": s, "n": len(v), "mean": v.mean(), "sd": v.std(ddof=1), "min": v.min(),
                     **{f"q{int(q*100):02d}": v.quantile(q) for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)},
                     "max": v.max(), "n_zero": int((v == 0).sum())})
    D = pd.DataFrame(dist)
    rows = []
    for t in GRID:
        r = {"criterion": f"nDCG@10 < {t:.2f}", "threshold": t}
        ok = True
        for s in FOLDS:
            pos = int((nd[s] < t).sum()); neg = len(nd[s]) - pos
            r[f"pos_{s}"], r[f"neg_{s}"] = pos, neg
            ok &= pos >= MIN_PER_CLASS and neg >= MIN_PER_CLASS
        r["feasible"] = ok
        rows.append(r)
    # anchored criterion: no grade>=2 passage in top 10
    r = {"criterion": "no grade>=2 passage in top 10", "threshold": np.nan}
    ok = True
    for s in FOLDS:
        q = common.load_qrels(s); run = m07.load_run(s)
        run = run[run["qid"].isin(nd[s].index)].sort_values(["qid", "rank"])
        top = run.groupby("qid").head(10)
        hit = top.assign(rel=[q[a].get(d, 0) >= common.GRADE_RELEVANT_THRESHOLD for a, d in zip(top["qid"], top["docid"])]) \
                 .groupby("qid")["rel"].any()
        pos = int((~hit).sum()); neg = len(hit) - pos
        r[f"pos_{s}"], r[f"neg_{s}"] = pos, neg
        ok &= pos >= MIN_PER_CLASS and neg >= MIN_PER_CLASS
    r["feasible"] = ok
    rows.append(r)
    C = pd.DataFrame(rows)
    D.to_csv(OUT / "tab_ndcg10_distribution.csv", index=False)
    C.to_csv(OUT / "tab_threshold_counts.csv", index=False)
    pd.set_option("display.width", 220)
    print(D.round(3).to_string(index=False)); print(); print(C.to_string(index=False))
    for s in FOLDS:
        print(s, "sorted nDCG@10:", " ".join(f"{x:.2f}" for x in sorted(nd[s])))


if __name__ == "__main__":
    main()
