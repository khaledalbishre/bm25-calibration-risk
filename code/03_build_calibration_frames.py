"""
Phase 2 (part 1) — build document-level calibration frames.

For each query, each judged passage in the top-1000 run ->
(qid, docid, rank, bm25_score, grade, label = 1 if grade >= GRADE_RELEVANT_THRESHOLD else 0).
Primary analysis uses judged pairs only.

Usage:
    python code/03_build_calibration_frames.py
Reads:
    runs/bm25_dl19.trec, runs/bm25_dl20.trec
Writes:
    data/calib_frame_dl19.parquet
    data/calib_frame_dl20.parquet
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")

import pandas as pd


def load_run(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            qid, _, docid, rank, score, tag = line.strip().split()
            rows.append({"qid": qid, "docid": docid, "rank": int(rank), "bm25_score": float(score)})
    return pd.DataFrame(rows)


def build_frame(split: str) -> pd.DataFrame:
    qrels = common.load_qrels(split)  # local TREC qrels files
    run_df = load_run(common.RUNS_DIR / f"bm25_{split}.trec")

    def grade_of(row):
        return qrels.get(row["qid"], {}).get(row["docid"], None)

    run_df["grade"] = run_df.apply(grade_of, axis=1)
    judged = run_df.dropna(subset=["grade"]).copy()
    judged["grade"] = judged["grade"].astype(int)
    judged["label"] = (judged["grade"] >= common.GRADE_RELEVANT_THRESHOLD).astype(int)
    judged["split"] = split
    return judged[["split", "qid", "docid", "rank", "bm25_score", "grade", "label"]]


def main():
    for split in ("dl19", "dl20"):
        frame = build_frame(split)
        out_path = common.DATA_DIR / f"calib_frame_{split}.parquet"
        frame.to_parquet(out_path, index=False)
        print(
            f"{split}: {len(frame)} judged (qid, docid) pairs across "
            f"{frame['qid'].nunique()} queries -> {out_path}"
        )


if __name__ == "__main__":
    main()
