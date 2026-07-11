# Hybrid Evaluation for Table-Text RAG

This repository contains the public code package and replication materials for a master's thesis on diagnostic evaluation for HybridQA and table-text retrieval-augmented generation systems.

The project studies whether evaluation should go beyond final answer matching and include diagnostic signals for retrieval, evidence-path consistency, modality-aware factual grounding, modality attribution, complexity-aware aggregation, and robustness under evidence perturbation.

## Repository contents

```text
config.py                 # central paths, model names, and experiment settings
requirements.txt          # Python dependencies
src/                      # metric, retrieval, pipeline, and utility code
scripts/                  # data processing, evaluation, and analysis scripts
results/                  # saved JSON summaries used in the thesis analysis
data/                     # placeholder folders and data instructions only
```

This public package intentionally excludes raw HybridQA data files, large generated JSONL outputs, local logs, API credentials, and environment-specific files.

## Dataset

The experiments use the public HybridQA dataset from the official HybridQA repository:

```text
https://github.com/wenhuchen/HybridQA
```

The raw dataset is not redistributed in this repository because of size and licensing considerations. Please download the released data from the official repository and place the required files under:

```text
data/raw/
```

## Environment

The code was developed with Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```

API-dependent scripts use Alibaba Cloud Bailian / DashScope-compatible Qwen endpoints through the OpenAI-compatible client. Set the following environment variables before running scripts that call the judge model or the RAG model:

```bash
BAILIAN_API_KEY=your_api_key
BAILIAN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
JUDGE_MODEL=qwen-max
RAG_MODEL=qwen-max
```

API keys should be stored locally as environment variables or in a local `.env` file. Do not commit credentials to the repository.

## Main scripts

The scripts are numbered approximately in the order used during the project. API-dependent scripts require `BAILIAN_API_KEY`; analysis scripts mostly operate on saved files under `results/`.

| Script | Purpose |
|---|---|
| `scripts/01_download_data.py` | Download or prepare HybridQA data. |
| `scripts/01b_build_unified_data.py` | Build unified local data files. |
| `scripts/02_build_perturbations.py` | Generate evidence perturbation conditions. |
| `scripts/03_run_baseline.py` | Run the baseline HybridQA evaluation pipeline. |
| `scripts/04_run_perturbation.py` | Evaluate the pipeline under evidence perturbations. |
| `scripts/05_stratified_analysis.py` | Compute complexity-stratified results and CWRS. |
| `scripts/06_human_correlation.py` | Compute metric correlations with LLM-simulated judgment scores. |
| `scripts/08_comparison_metrics.py` | Compute comparison metrics such as BERTScore proxy, FActScore, RAGAS variants, RGB, and G-Eval variants. |
| `scripts/10_routing_ablation.py` | Compare modality-routed H-FAct with a no-routing FActScore-style baseline. |
| `scripts/12_bootstrap_ci.py` | Estimate confidence intervals by bootstrap resampling. |
| `scripts/13_multiple_comparison.py` | Apply multiple-comparison tests and the canonical Williams comparison. |
| `scripts/14_complementarity_v3.py` | Compute the final H-FAct v3 complementarity analysis. |

## Saved result summaries

The public repository includes JSON summary files that support the reported analysis, including:

```text
results/baseline/summary.json
results/perturbation/summary.json
results/stratified/summary.json
results/correlation/summary.json
results/correlation_v3/summary_v3.json
results/full_v3/summary_full_v3.json
results/ablation/routing_ablation.json
results/ablation/complementarity_v3.json
results/ablation/significance_tests.json
results/comparison/multiple_comparison.json
results/comparison/williams_canonical.json
results/bootstrap/
```

Large JSONL files with full predictions, raw contexts, or intermediate annotated samples are excluded from the public package. They can be regenerated from the scripts if the dataset and API credentials are available.

## Scope notes

- The judgment signal used in the thesis is LLM-simulated and should be understood as a controlled proxy rather than external human annotation.
- H-FAct v3 is the primary reported version for the main validation study.
- Traditional answer-level metrics such as EM, F1, and BERTScore are not replaced by this framework; Hybrid Evaluation adds diagnostic grounding-oriented signals.
- Results should be interpreted within the controlled HybridQA setup used in the thesis.

## Security and privacy

This public package excludes:

```text
.env
API keys
local logs
raw dataset files
large generated JSONL outputs
local absolute-path configuration
```

Before making the repository public, verify that no credentials or private files have been committed in the repository history.

