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
    parser = argparse.ArgumentParser(description="Medallion pipeline orchestration — Bronze → Silver → Gold → Report")
    parser.add_argument("--files", nargs="*", default=["data/landing/sales_data.csv"])
    parser.add_argument("--run-id", default="local-run")
    parser.add_argument("--task", default="Profile and sample the retail CSV dataset.")
    parser.add_argument("--business-intent", default="", help="Business question to drive transformations")
    parser.add_argument("--skip-approval", action="store_true", help="Skip HITL approval gates (not recommended for production)")
    args = parser.parse_args()

    file_paths = [Path(p) for p in args.files]

    print("\n" + "="*70)
    print("MEDALLION PIPELINE - End-to-End Data Transformation")
    print("="*70)
    print(f"Run ID: {args.run_id}")
    print(f"Input Files: {[str(p) for p in file_paths]}")
    print(f"Business Intent: {args.business_intent or '(none)'}")
    print("="*70 + "\n")

    # Phase 1: Profile
    print("[1/8] PROFILING RAW DATA...")
    profile_path = profile_multiple_datasets(file_paths, args.run_id, args.task)
    print(f"  ✅ Profile saved to {profile_path}\n")

    # Phase 2: Bronze STTM
    print("[2/8] BRONZE LAYER - Generating STTM...")
    bronze_sttm = generate_bronze_sttm(file_paths)
    print(f"  STTM: {bronze_sttm}")
    if not args.skip_approval:
        approval = input("  ➤ Approve Bronze STTM? (y/n): ").strip().lower()
        if approval != "y":
            print("  ❌ Bronze STTM rejected. Exiting.")
            return
    print("  ✅ Bronze STTM approved\n")

    # Phase 3: Bronze Ingest
    print("[3/8] BRONZE LAYER - Ingesting data...")
    bronze_outputs = bronze_ingest(file_paths, output_dir="data/bronze_layer")
    print(f"  ✅ Bronze outputs: {bronze_outputs}\n")

    # Phase 4: Silver STTM
    print("[4/8] SILVER LAYER - Generating STTM...")
    silver_sttm = generate_silver_sttm(file_paths)
    print(f"  STTM: {silver_sttm}")
    if not args.skip_approval:
        approval = input("  ➤ Approve Silver STTM? (y/n): ").strip().lower()
        if approval != "y":
            print("  ❌ Silver STTM rejected. Exiting.")
            return
    print("  ✅ Silver STTM approved\n")

    # Phase 5: Silver Clean (reads from Bronze outputs)
    print("[5/8] SILVER LAYER - Cleaning and standardizing...")
    silver_outputs = silver_clean(bronze_outputs, output_dir="data/silver_layer", business_intent=args.business_intent)
    print(f"  ✅ Silver outputs: {silver_outputs}\n")

    # Phase 6: Gold STTM
    print("[6/8] GOLD LAYER - Generating STTM...")
    gold_sttm = generate_gold_sttm(file_paths)
    print(f"  STTM: {gold_sttm}")
    if not args.skip_approval:
        approval = input("  ➤ Approve Gold STTM? (y/n): ").strip().lower()
        if approval != "y":
            print("  ❌ Gold STTM rejected. Exiting.")
            return
    print("  ✅ Gold STTM approved\n")

    # Phase 7: Gold Aggregate (reads from Silver outputs)
    print("[7/8] GOLD LAYER - Aggregating and creating KPIs...")
    gold_outputs = gold_aggregate(silver_outputs, output_dir="data/gold_layer", business_intent=args.business_intent)
    print(f"  ✅ Gold outputs: {gold_outputs}\n")

    # Phase 8: Report (reads from Gold outputs)
    print("[8/8] REPORTING - Generating executive report...")
    insight = create_report_insight(gold_outputs, report_path="reports", business_question=args.business_intent, source_files=[str(p) for p in file_paths])
    print(f"  ✅ Report generated: {insight.get('source')}\n")

    print("="*70)
    print("✅ PIPELINE COMPLETE")
    print("="*70)
    print(f"Profile:       {profile_path}")
    print(f"Bronze STTM:   {json.dumps(bronze_sttm, indent=2)}")
    print(f"Bronze Data:   {bronze_outputs}")
    print(f"Silver STTM:   {json.dumps(silver_sttm, indent=2)}")
    print(f"Silver Data:   {silver_outputs}")
    print(f"Gold STTM:     {json.dumps(gold_sttm, indent=2)}")
    print(f"Gold Data:     {gold_outputs}")
    print(f"Report:        {insight.get('source')}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
