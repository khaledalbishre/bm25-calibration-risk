# Regenerates every result table and figure of the article (Windows).
#
# Requires (see README.md): the Pyserini prebuilt msmarco-v1-passage index
# and, in data\, msmarco-test2019-queries.tsv, msmarco-test2020-queries.tsv,
# 2019qrels-pass.txt and 2020qrels-pass.txt. With -FromRuns, retrieval is
# skipped and only the two qrels files are needed.
#
# Usage (from the repository root, inside the activated environment):
#   powershell -ExecutionPolicy Bypass -File code\run_all.ps1 [-FromRuns]
param([switch]$FromRuns)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))

# Pyserini constructs an OpenAI client at import time and raises without a
# key, although this pipeline never calls OpenAI.
if (-not $env:OPENAI_API_KEY) { $env:OPENAI_API_KEY = "dummy-not-used" }

$steps = @(
    @("unit tests (synthetic data)", "tests_synthetic.py"),
    @("BM25 retrieval", "01_retrieve.py"),
    @("evaluation and published-baseline checkpoint", "02_evaluate_checkpoint1.py"),
    @("calibration frames", "03_build_calibration_frames.py"),
    @("single-direction calibrators (superseded table)", "04_fit_calibrators.py"),
    @("pooled risk analysis (superseded table)", "05_risk_signal.py"),
    @("diagnostics: base rates, norm+ variants, sign checks", "06_diagnostics.py"),
    @("fold-aware risk detection and depth-matched diagnostic (Tables 4-6)", "07_fold_aware_risk.py"),
    @("two-fold calibration and Brier bootstrap (Table 3)", "10_twofold_calibration.py"),
    @("figures (checked against the result tables)", "08_final_figures.py")
)
foreach ($s in $steps) {
    if ($FromRuns -and $s[1] -eq "01_retrieve.py") { continue }
    Write-Host "=== $($s[0]) ==="
    python "code\$($s[1])"
    if ($LASTEXITCODE -ne 0) { throw "$($s[1]) failed" }
}
Write-Host "=== DONE. See results\ ==="
