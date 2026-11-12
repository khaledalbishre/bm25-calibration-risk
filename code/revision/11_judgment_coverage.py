"""
Judgment coverage of the BM25 runs (Section 4.2) at depths 10 and 100 (1000 for
context). Descriptive only; no model is fitted.

For each judged query (43 DL19, 54 DL20) and depth k: the fraction of the
top-k passages that have any relevance judgment, and the fraction judged
relevant (grade >= 2) / judged non-relevant (grade < 2).

Reads  runs/bm25_{dl19,dl20}.trec, data/20{19,20}qrels-pass.txt
Writes results/revision/tab_judgment_coverage.csv          (summary)
       results/revision/tab_judgment_coverage_perquery.csv (per query)
"""
import numpy as np
import pandas as pd
from revision_common import common, m07, OUT, FOLDS

DEPTHS = (10, 100, 1000)


def main():
    per_q, summ = [], []
    for s in FOLDS:
        qrels = common.load_qrels(s)
        run = m07.load_run(s)
        run = run[run["qid"].isin(qrels.keys())].sort_values(["qid", "rank"])
        for qid, g in run.groupby("qid"):
            j = qrels[qid]
            grades = np.array([j.get(d, -1) for d in g["docid"]])
            for k in DEPTHS:
                gk = grades[:k]
                per_q.append({"fold": s, "qid": qid, "depth": k, "n": len(gk),
                              "frac_judged": float((gk >= 0).mean()),
                              "frac_judged_rel": float((gk >= common.GRADE_RELEVANT_THRESHOLD).mean()),
                              "frac_judged_nonrel": float(((gk >= 0) & (gk < common.GRADE_RELEVANT_THRESHOLD)).mean())})
    P = pd.DataFrame(per_q)
    for (s, k), g in P.groupby(["fold", "depth"]):
        summ.append({"fold": s, "depth": k, "n_queries": len(g),
                     "pooled_frac_judged": float((g["frac_judged"] * g["n"]).sum() / g["n"].sum()),
                     "mean_frac_judged": g["frac_judged"].mean(), "median_frac_judged": g["frac_judged"].median(),
                     "min_frac_judged": g["frac_judged"].min(), "max_frac_judged": g["frac_judged"].max(),
                     "n_queries_fully_judged": int((g["frac_judged"] == 1).sum()),
                     "mean_frac_judged_rel": g["frac_judged_rel"].mean(),
                     "mean_frac_judged_nonrel": g["frac_judged_nonrel"].mean(),
                     "mean_frac_unjudged": 1 - g["frac_judged"].mean()})
    S = pd.DataFrame(summ)
    P.to_csv(OUT / "tab_judgment_coverage_perquery.csv", index=False)
    S.to_csv(OUT / "tab_judgment_coverage.csv", index=False)
    pd.set_option("display.width", 220)
    print(S.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
