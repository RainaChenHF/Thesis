"""
Script 02 – Build the five perturbed datasets from HybridQA dev.

Outputs (data/perturbed/):
  golden.jsonl
  semi_golden.jsonl
  noise.jsonl
  counterfactual.jsonl
  missing.jsonl
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.perturbation import (
    load_dev, load_tables, load_passages,
    build_perturbed, save_perturbed,
)


def main():
    print("=== Build Perturbed Datasets ===")
    dev      = load_dev()
    tables   = load_tables()
    passages = load_passages()
    print(f"  dev={len(dev)}  tables={len(tables)}  passages={len(passages)}")

    buckets = build_perturbed(dev, tables, passages)
    save_perturbed(buckets)
    print("Done.")


if __name__ == "__main__":
    main()
