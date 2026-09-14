from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bronze import bronze_ingest
from agents.gold import gold_aggregate
from agents.orchestrator import run_pipeline_phase
from agents.profiler import profile_multiple_datasets
from agents.reporter import create_report_insight
from agents.silver import silver_clean
from agents.sttm import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm


def main() -> None:
    parser = argparse.ArgumentParser(description="Medallion pipeline example entry point")
    parser.add_argument("--files", nargs="*", default=["data/landing/sales_data.csv"])
    parser.add_argument("--run-id", default="local-run")
    parser.add_argument("--task", default="Profile and sample the retail CSV dataset.")
    args = parser.parse_args()

    file_paths = [Path(p) for p in args.files]
    profile_path = profile_multiple_datasets(file_paths, args.run_id, args.task)

    bronze_sttm = generate_bronze_sttm(file_paths)
    silver_sttm = generate_silver_sttm(file_paths)
    gold_sttm = generate_gold_sttm(file_paths)

    phase_state = run_pipeline_phase("phase1", [str(p) for p in file_paths], args.run_id)
    bronze_outputs = bronze_ingest(file_paths, output_dir="data/bronze_layer")
    silver_outputs = silver_clean(file_paths, output_dir="data/silver_layer")
    gold_outputs = gold_aggregate(file_paths, output_dir="data/gold_layer")

    insight = create_report_insight(gold_outputs[0], report_path="reports")

    print("profile_path:", profile_path)
    print("bronze_sttm:", bronze_sttm)
    print("silver_sttm:", silver_sttm)
    print("gold_sttm:", gold_sttm)
    print("phase_state:", phase_state.status)
    print("bronze_outputs:", bronze_outputs)
    print("silver_outputs:", silver_outputs)
    print("gold_outputs:", gold_outputs)
    print("report_insights:", insight)


if __name__ == "__main__":
    main()
