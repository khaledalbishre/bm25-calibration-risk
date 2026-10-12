"""
Phase 1 — BM25 top-1000 retrieval on TREC DL 2019 / 2020, default k1/b.

Data sources: local files only.
  - Index: the standard Pyserini prebuilt `msmarco-v1-passage` Lucene
    index, opened with LuceneSearcher(<path>).
  - Topics: official msmarco-test2019/2020-queries.tsv, restricted to
    the judged qids (those present in 2019qrels-pass.txt /
    2020qrels-pass.txt): 43 for DL19, 54 for DL20.

Usage:
    python code/01_retrieve.py
Writes:
    runs/bm25_dl19.trec
    runs/bm25_dl20.trec
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

common = import_module("00_common")

from pyserini.search.lucene import LuceneSearcher


def run_bm25(searcher, topics: dict, out_path: Path, run_tag: str, k: int = common.TOP_K_RETRIEVAL):
    # newline="\n" so TREC run files are LF-terminated even on Windows.
    with open(out_path, "w", newline="\n") as fh:
        for qid in sorted(topics, key=lambda x: int(x)):
            query_text = topics[qid]
            hits = searcher.search(query_text, k=k)
            for rank, hit in enumerate(hits, start=1):
                fh.write(f"{qid} Q0 {hit.docid} {rank} {hit.score:.6f} {run_tag}\n")
    print(f"wrote {out_path} ({len(topics)} queries, top-{k})")


def main():
    assert common.INDEX_DIR.is_dir(), f"index not found at {common.INDEX_DIR}"
    searcher = LuceneSearcher(str(common.INDEX_DIR))
    # Explicit default params, recorded even though they are the library
    # defaults, per "no tuning" + "record exact versions" discipline.
    searcher.set_bm25(common.BM25_K1_DEFAULT, common.BM25_B_DEFAULT)
    print(f"index: {common.INDEX_DIR} ({searcher.num_docs} docs)")

    for split in ("dl19", "dl20"):
        topics = common.load_judged_queries(split)
        out_path = common.RUNS_DIR / f"bm25_{split}.trec"
        run_bm25(searcher, topics, out_path, run_tag=f"bm25_default_{split}")


if __name__ == "__main__":
    main()
