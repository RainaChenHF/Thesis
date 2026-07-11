"""
Central configuration for HybridQA Evaluation Project.
All API keys and paths are sourced from environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # loads .env if present

# â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data"
RAW_DIR     = DATA_DIR / "raw"
PERT_DIR    = DATA_DIR / "perturbed"
ANN_DIR     = DATA_DIR / "annotations"
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = ROOT / "logs"

for d in [RAW_DIR, PERT_DIR, ANN_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# â”€â”€ Qwen API (é˜¿é‡Œäº‘ç™¾ç‚¼) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
QWEN_API_KEY  = os.getenv("BAILIAN_API_KEY", "")
QWEN_BASE_URL = os.getenv("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
JUDGE_MODEL   = os.getenv("JUDGE_MODEL", "qwen-max")      # LLM-as-Judge
RAG_MODEL     = os.getenv("RAG_MODEL",   "qwen-max")      # RAG answer generation

if not QWEN_API_KEY:
    import warnings
    warnings.warn(
        "BAILIAN_API_KEY not set. API calls (H-FAct, MAA, HR-P, RAG generation) will fail. "
        "Non-API scripts (download, build data, stratification, report) run fine without it.",
        stacklevel=2,
    )

# â”€â”€ Experiment settings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RETRIEVAL_TOP_K      = 3       # JHit@K
PERTURBATION_SUBSET  = 500     # rows sampled per perturbation experiment
HUMAN_CORR_SUBSET    = 100     # rows for human correlation study
RANDOM_SEED          = 42

# HybridQA dataset source
HYBRIDQA_REPO = "https://github.com/wenhuchen/HybridQA"  # official dataset/code repository

