# Calibration Quality and Query-Level Risk Detection in BM25 Retrieval

Reproducibility package for the article *Calibration Quality and
Query-Level Risk Detection in BM25 Retrieval* by Khaled Albishre
(Department of Computer Science, University College of Al Jamoum, Umm
Al-Qura University), submitted to *Future Internet* (MDPI).

The package contains the analysis code, the BM25 run files, the
judged-pair calibration frames, every result table, the four
publication figures with their plotted values, and the provenance file
that links every number in the manuscript to its source. It regenerates
every table and figure of the article.

## Contents

| Path | Content |
|---|---|
| `code/` | Pipeline scripts `00`–`08`, `10`, synthetic unit tests, `run_all.sh` / `run_all.ps1` |
| `runs/bm25_dl19.trec`, `runs/bm25_dl20.trec` | BM25 top-1000 runs (TREC run format; passage ids and scores only) |
| `data/calib_frame_dl19.parquet`, `data/calib_frame_dl20.parquet` | Judged pairs of the runs: query id, passage id, rank, BM25 score, relevance grade, binary label (5,066 and 5,904 pairs) |
| `results/tab_*.csv` | All result tables |
| `results/figures_final/` | Figures 1–4 (PDF and PNG) and `figure_values.csv` (every plotted value with its source table and check) |
| `results/number_provenance.csv` | Source, lookup key or formula, rounding rule and verification status of every number in the manuscript |
| `requirements.txt`, `requirements-lock.txt` | Pinned direct dependencies and the complete pinned environment |

No passage text or query text is included. The relevance grades in the
calibration frames are those of the public TREC Deep Learning qrels.

## Obtaining the benchmark data

The benchmark data are not redistributed here. Download them from their
official sources and place the four TREC files in `data/`.

**MS MARCO passage collection.** The experiments use the collection
through the Pyserini prebuilt Lucene index (next item), so the raw
collection is needed only to build the index yourself. It is available
from the MS MARCO website, <https://microsoft.github.io/msmarco/>
(`collection.tar.gz`), under the MS MARCO terms of use.

**Pyserini prebuilt index `msmarco-v1-passage`** (Lucene, version
`20221004.252b5e`). With the environment below installed, run once

```
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')"
```

which downloads the index to `~/.cache/pyserini/indexes/`, where
`code/00_common.py` finds it. To use an index stored elsewhere, set the
environment variable `BM25CAL_INDEX_DIR` to its directory.

**TREC Deep Learning 2019 (passage task).** Track page:
<https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019>

- topics: <https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz>
  (decompress to `data/msmarco-test2019-queries.tsv`)
- judgments: <https://trec.nist.gov/data/deep/2019qrels-pass.txt>
  (save as `data/2019qrels-pass.txt`)

**TREC Deep Learning 2020 (passage task).** Track page:
<https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020>

- topics: <https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2020-queries.tsv.gz>
  (decompress to `data/msmarco-test2020-queries.tsv`)
- judgments: <https://trec.nist.gov/data/deep/2020qrels-pass.txt>
  (save as `data/2020qrels-pass.txt`)

SHA-256 checksums of the files used:

```
fabc43e4bbcb0ba385c647d38d54a1ca08961570c4d654739644df47c4c8b4e0  msmarco-test2019-queries.tsv
726cb4018b85a335fcbe09004df7f084ec1bf20531961d9af7b0c71dd609e69b  msmarco-test2020-queries.tsv
8a1f10d550732e4cd91d7fc49846a3784de4040972f583e69285a88f3c5fee92  2019qrels-pass.txt
60d4c34561f9687f8f73e0a752ba80ab80159b3e9c1bedd0a351f747ed6f5684  2020qrels-pass.txt
```

Only the judged queries are used: 43 for DL19 and 54 for DL20.

## Environment

Python 3.12 (3.12.10 used) and Java 21 (Eclipse Temurin 21.0.10 used,
with `JAVA_HOME` set) for Pyserini/Anserini. Main packages: Pyserini
1.6.0 (bundles Anserini 1.7.1, Lucene 9.9.1), pytrec_eval-terrier
0.5.10, scikit-learn 1.9.0, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0,
pyarrow 25.0.1, matplotlib 3.11.1.

```
python3.12 -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # or requirements-lock.txt for the exact full set
```

Pyserini constructs an OpenAI client at import time and raises if
`OPENAI_API_KEY` is unset, although this pipeline never calls OpenAI.
The scripts set a dummy value automatically.

## Running

From the repository root:

```
bash code/run_all.sh                 # full pipeline, including BM25 retrieval
bash code/run_all.sh --from-runs     # skip retrieval; start from the shipped run files
```

On Windows: `powershell -ExecutionPolicy Bypass -File code\run_all.ps1`
(add `-FromRuns` to skip retrieval). The full pipeline needs the index
and the four TREC files; `--from-runs` needs only the two qrels files.
It takes a few minutes on a laptop CPU.

| Step | Script | Output | Article |
|---|---|---|---|
| unit tests | `tests_synthetic.py` | checks of ECE, Brier, AUROC and bootstrap helpers on hand-built arrays | — |
| 1 | `01_retrieve.py` | `runs/bm25_dl{19,20}.trec` (BM25, k1 = 0.9, b = 0.4, top 1000) | Section 4.1 |
| 2 | `02_evaluate_checkpoint1.py` | `tab_retrieval_eval.csv`; stops if nDCG@10 differs from the published Pyserini baseline by more than 0.005 | Table 2 |
| 3 | `03_build_calibration_frames.py` | `data/calib_frame_*.parquet` | Section 4.2 |
| 4 | `04_fit_calibrators.py` | `tab_calibration_dl20.csv` (single-direction table, superseded by step 8) | — |
| 5 | `05_risk_signal.py` | `tab_risk_auroc.csv` (pooled AUROC, superseded by step 7) | — |
| 6 | `06_diagnostics.py` | `tab_diagnostics.csv`; base rates, norm+ rows | Section 5.2 |
| 7 | `07_fold_aware_risk.py` | `tab_risk_foldaware.csv`, `tab_risk_contrasts.csv`, `tab_fold_summary.csv` | Tables 4–6 |
| 8 | `10_twofold_calibration.py` | `tab_calibration_twofold.csv`, `tab_calibration_pooled_oof.csv`, `tab_calibration_brier_ci.csv`, `tab_calibration_brier_contrasts.csv` | Table 3, Section 5.2 |
| 9 | `08_final_figures.py` | `results/figures_final/` | Figures 1–4 |

`08_final_figures.py` recomputes or reads every plotted value, checks it
against the result tables, and stops on any mismatch.

## Determinism

BM25 retrieval, Platt scaling (unregularized logistic regression,
lbfgs) and isotonic regression are deterministic. The only random
component is the bootstrap (2000 resamples, `numpy.random.default_rng`
with seed `20260816`, set in `code/00_common.py`). A rerun in the
environment above reproduces the run files byte for byte and every
table and plotted value exactly.

`LogisticRegression(penalty=None)` emits a deprecation warning on
scikit-learn 1.8 and later and is removed in 1.10; use `C=np.inf` when
upgrading.

## License

Code: MIT License (see `LICENSE`). Result tables, figures, run files
and calibration frames are released under CC BY 4.0. The MS MARCO and
TREC Deep Learning data remain subject to their own terms.
