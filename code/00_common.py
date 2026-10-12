"""
Shared constants and helpers.

Every other script in code/ imports from here so that locked design
decisions are defined exactly once.

IMPORTANT: importing pyserini transitively constructs an openai.OpenAI()
client at import time (pyserini/encode/_openai.py) and raises if
OPENAI_API_KEY is unset, even though this project never calls OpenAI.
We set a dummy value here before any pyserini import happens anywhere
in the pipeline.
"""
import os

os.environ.setdefault("OPENAI_API_KEY", "dummy-not-used")

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"

for d in (DATA_DIR, RUNS_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Fixed design ----

# Inputs are the official TREC DL topic and qrels files in data/ and the
# standard Pyserini prebuilt index, opened directly with
# LuceneSearcher(<path>). INDEX_DIR resolution order:
#   1. BM25CAL_INDEX_DIR environment variable, if set;
#   2. the default Pyserini cache location below, if it exists;
#   3. data/index/<name> relative to the repository root (fallback).
PREBUILT_INDEX = "msmarco-v1-passage"  # provenance label only

_CACHE_INDEX = (
    Path.home() / ".cache" / "pyserini" / "indexes"
    / "lucene-inverted.msmarco-v1-passage.20221004.252b5e.678876e8c99a89933d553609a0fd8793"
)
_FALLBACK_INDEX = DATA_DIR / "index" / "lucene-inverted.msmarco-v1-passage.20221004.252b5e"

if os.environ.get("BM25CAL_INDEX_DIR"):
    INDEX_DIR = Path(os.environ["BM25CAL_INDEX_DIR"])
elif _CACHE_INDEX.is_dir():
    INDEX_DIR = _CACHE_INDEX
else:
    INDEX_DIR = _FALLBACK_INDEX

QUERY_FILES = {
    "dl19": DATA_DIR / "msmarco-test2019-queries.tsv",
    "dl20": DATA_DIR / "msmarco-test2020-queries.tsv",
}
QRELS_FILES = {
    "dl19": DATA_DIR / "2019qrels-pass.txt",
    "dl20": DATA_DIR / "2020qrels-pass.txt",
}


def load_qrels(split: str) -> dict:
    """TREC qrels format: qid iter docid grade -> {qid: {docid: grade}}."""
    qrels = {}
    with open(QRELS_FILES[split]) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 4:
                continue
            qid, _, docid, grade = parts
            qrels.setdefault(qid, {})[docid] = int(grade)
    return qrels


def load_judged_queries(split: str) -> dict:
    """{qid: query_text}, restricted to qids present in the qrels
    (43 for DL19, 54 for DL20 out of 200 in each query file)."""
    judged_qids = set(load_qrels(split).keys())
    topics = {}
    with open(QUERY_FILES[split], encoding="utf-8") as fh:
        for line in fh:
            qid, text = line.rstrip("\n").split("\t", 1)
            if qid in judged_qids:
                topics[qid] = text
    return topics


TOP_K_RETRIEVAL = 1000
BM25_K1_DEFAULT = 0.9  # Anserini/Lucene default; recorded, not tuned
BM25_B_DEFAULT = 0.4  # Anserini/Lucene default; recorded, not tuned

GRADE_RELEVANT_THRESHOLD = 2  # label = 1 if grade >= 2 else 0

# Checkpoint 1: published Pyserini regression-test baselines for BM25
# default k1/b (row 1a, "BM25 (Anserini default k1/b)" / Ma et al. SIGIR
# 2022 provenance) on msmarco-v1-passage, from the live Pyserini
# reproduction matrix. VERIFIED via WebFetch against
# https://castorini.github.io/pyserini/2cr/msmarco-v1-passage.html on
# 2026-08-16.
PUBLISHED_BASELINES = {
    "dl19": {"ndcg@10": 0.5058, "recall@1000": 0.7501},
    "dl20": {"ndcg@10": 0.4796, "recall@1000": 0.7863},
}
CHECKPOINT1_TOLERANCE = 0.005

# QRC (Phase 3, locked)
QRC_K = 10  # top-10, fixed a priori, no sweep

# Dispersion baseline depth (CV@100, an NQC-style score); 100 is the
# standard value used in the original NQC paper and most QPP follow-ups.
NQC_K = 100

# Risk targets
RISK_TARGET_PRIMARY = "below_median_ndcg10"
RISK_TARGET_SECONDARY = "ndcg10_eq_0"

# Bootstrap
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260816  # fixed seed = date of this run, for reproducibility

RANDOM_SEED = 20260816
