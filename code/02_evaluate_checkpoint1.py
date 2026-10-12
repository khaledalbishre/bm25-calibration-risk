"""
Phase 1 — evaluate BM25 runs (nDCG@10, Recall@1000) and enforce
CHECKPOINT 1: |our nDCG@10 - published Pyserini baseline| <= 0.005.

Data sources: local qrels files only (2019qrels-pass.txt /
2020qrels-pass.txt).

TREC DL evaluation convention (matches the published Pyserini repro
matrix, which uses trec_eval with -l 2 for Recall):
  - nDCG@10 uses the graded qrels as-is.
  - Recall@1000 binarizes at grade >= 2 (relevance_level=2), so grade-1
    ("related") passages do not count as relevant.

Usage:
    python code/02_evaluate_checkpoint1.py
Reads:
    runs/bm25_dl19.trec, runs/bm25_dl20.trec, data/*qrels*.txt
Writes:
    results/tab_retrieval_eval.csv
    results/checkpoint1_failure.md (only on checkpoint failure)
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")

import pandas as pd
import pytrec_eval


def load_run(path: Path) -> dict:
    run = {}
    with open(path) as fh:
        for line in fh:
            qid, _, docid, rank, score, tag = line.strip().split()
            run.setdefault(qid, {})[docid] = float(score)
    return run


def evaluate(split: str) -> dict:
    qrels = common.load_qrels(split)
    run = load_run(common.RUNS_DIR / f"bm25_{split}.trec")

    # nDCG@10 on graded qrels (default relevance_level=1 does not affect nDCG).
    ev_ndcg = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut_10"})
    pq_ndcg = ev_ndcg.evaluate(run)
    # Recall@1000 with TREC DL binarization: grade >= 2 counts as relevant.
    ev_rec = pytrec_eval.RelevanceEvaluator(qrels, {"recall_1000"},
                                            relevance_level=common.GRADE_RELEVANT_THRESHOLD)
    pq_rec = ev_rec.evaluate(run)

    ndcg10 = sum(v["ndcg_cut_10"] for v in pq_ndcg.values()) / len(pq_ndcg)
    recall1000 = sum(v["recall_1000"] for v in pq_rec.values()) / len(pq_rec)
    return {
        "split": split,
        "n_queries": len(pq_ndcg),
        "ndcg@10": ndcg10,
        "recall@1000": recall1000,
    }


def checkpoint1(results: dict) -> bool:
    ok = True
    lines = []
    for split, res in results.items():
        published = common.PUBLISHED_BASELINES[split]["ndcg@10"]
        diff = abs(res["ndcg@10"] - published)
        passed = diff <= common.CHECKPOINT1_TOLERANCE
        ok = ok and passed
        lines.append(
            f"{split}: ndcg@10 ours={res['ndcg@10']:.4f} published={published:.4f} "
            f"diff={diff:.4f} tolerance={common.CHECKPOINT1_TOLERANCE} "
            f"{'PASS' if passed else 'FAIL'}  "
            f"[informational only, not gating] recall@1000 ours={res['recall@1000']:.4f} "
            f"published={common.PUBLISHED_BASELINES[split]['recall@1000']:.4f}"
        )
    print("\n".join(lines))
    if not ok:
        ts = datetime.now(timezone.utc).isoformat()
        with open(common.RESULTS_DIR / "checkpoint1_failure.md", "a") as fh:
            fh.write(f"\n## [{ts}] CHECKPOINT 1 FAILED\n\n")
            fh.write("\n".join(lines) + "\n")
            fh.write(
                "\nDiagnose: check index version/date, qrels source, "
                "binarization, and whether Recall@1000 pooling matches "
                "trec_eval convention before proceeding. Do not use these "
                "numbers in the paper until this is resolved.\n"
            )
    return ok


def main():
    results = {split: evaluate(split) for split in ("dl19", "dl20")}

    df = pd.DataFrame(list(results.values()))
    df.to_csv(common.RESULTS_DIR / "tab_retrieval_eval.csv", index=False)
    print(df.to_string(index=False))

    passed = checkpoint1(results)
    if not passed:
        print("CHECKPOINT 1 FAILED — see results/checkpoint1_failure.md. Not proceeding to Phase 2 on unverified numbers.")
        sys.exit(1)
    print("CHECKPOINT 1 PASSED.")


if __name__ == "__main__":
    main()
