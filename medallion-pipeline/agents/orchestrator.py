from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.state import PipelineState
from agents.profiler import profile_multiple_datasets
from agents.sttm import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from agents.bronze import bronze_ingest
from agents.silver import silver_clean
from agents.gold import gold_aggregate
from agents.reporter import create_report_insight


PHASES = [
    "User Intake",
    "Profile Raw Files",
    "Bronze STTM",
    "Bronze Agent",
    "Silver STTM",
    "Silver Agent",
    "Gold STTM",
    "Gold Agent",
    "Reporter Agent",
]


def _write_sttm(sttm_dict: dict, path: str) -> None:
    """Persist an STTM dict to JSON on disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sttm_dict, indent=2), encoding="utf-8")


def run_pipeline_phase(phase: str, file_paths: list[str], run_id: str = "") -> PipelineState:
    """Create a PipelineState snapshot for the requested Medallion workflow step."""
    state = PipelineState(run_id=run_id)
    state.status = phase
    state.uploaded_files = [str(Path(p)) for p in file_paths]
    return state


def run_stepwise_pipeline(file_paths: list[str], run_id: str = "") -> dict[str, Any]:
    """Return a phase-by-phase workflow plan that mirrors the UI orchestration."""
    ordered = [
        {"phase": i, "name": phase, "status": "pending", "run_id": run_id, "rows": None}
        for i, phase in enumerate(PHASES)
    ]
    return {
        "run_id": run_id,
        "phase_count": len(PHASES),
        "phases": ordered,
        "files": [str(Path(p)) for p in file_paths],
    }


def orchestrate_pipeline(
    file_paths: list[str],
    run_id: str = "",
    business_intent: str = "",
    skip_approval: bool = False,
    base_dir: str | Path = ".",
) -> dict[str, Any]:
    """Execute the full medallion pipeline end-to-end.

    Each STTM is written to disk and each agent reads from the previous layer:
      Landing CSVs → Bronze Parquet → Silver Parquet → Gold Parquet → Report
    """
    base_path = Path(base_dir)
    sttm_dir = base_path / "data" / "sttm"
    sttm_dir.mkdir(parents=True, exist_ok=True)

    state = PipelineState(run_id=run_id)
    state.business_intent = business_intent
    state.uploaded_files = file_paths

    try:
        # Phase 1: Profile raw files
        state.status = "Profile Raw Files"
        profile_path = profile_multiple_datasets(file_paths, run_id, "Profile retail dataset")
        state.profile_path = profile_path

        # Phase 2: Bronze STTM — analyze landing CSV schema
        state.status = "Bronze STTM"
        bronze_sttm = generate_bronze_sttm(file_paths)
        bronze_sttm_path = str(sttm_dir / f"bronze_sttm_{run_id}.json")
        _write_sttm(bronze_sttm, bronze_sttm_path)
        state.sttm_bronze_path = bronze_sttm_path
        state.hitl_approved = False

        if not skip_approval:
            state.status = "Bronze STTM (Awaiting Approval)"
        state.hitl_approved = True  # approved (CLI prompt handled by main.py)

        # Phase 3: Bronze ingest
        state.status = "Bronze Agent"
        bronze_outputs = bronze_ingest(file_paths, output_dir=str(base_path / "data" / "bronze_layer"))
        state.bronze_output_paths = bronze_outputs

        # Phase 4: Silver STTM — analyze actual Bronze Parquet schema
        state.status = "Silver STTM"
        silver_sttm = generate_silver_sttm(bronze_outputs, business_intent)
        silver_sttm_path = str(sttm_dir / f"silver_sttm_{run_id}.json")
        _write_sttm(silver_sttm, silver_sttm_path)
        state.sttm_silver_path = silver_sttm_path
        state.hitl_approved = False

        if not skip_approval:
            state.status = "Silver STTM (Awaiting Approval)"
        state.hitl_approved = True

        # Phase 5: Silver clean — reads from Bronze Parquet outputs
        state.status = "Silver Agent"
        silver_outputs = silver_clean(
            bronze_outputs,
            output_dir=str(base_path / "data" / "silver_layer"),
            business_intent=business_intent,
        )
        state.silver_output_paths = silver_outputs

        # Phase 6: Gold STTM — analyze actual Silver Parquet schema
        state.status = "Gold STTM"
        gold_sttm = generate_gold_sttm(silver_outputs, business_intent)
        gold_sttm_path = str(sttm_dir / f"gold_sttm_{run_id}.json")
        _write_sttm(gold_sttm, gold_sttm_path)
        state.sttm_gold_path = gold_sttm_path
        state.hitl_approved = False

        if not skip_approval:
            state.status = "Gold STTM (Awaiting Approval)"
        state.hitl_approved = True

        # Phase 7: Gold aggregate — reads from Silver Parquet outputs
        state.status = "Gold Agent"
        gold_outputs = gold_aggregate(
            silver_outputs,
            output_dir=str(base_path / "data" / "gold_layer"),
            business_intent=business_intent,
        )
        state.gold_output_paths = gold_outputs

        # Phase 8: Report — reads from Gold Parquet outputs
        state.status = "Reporter Agent"
        report = create_report_insight(
            gold_outputs,
            report_path=str(base_path / "reports"),
            business_question=business_intent,
            source_files=file_paths,
        )
        state.report_path = report.get("source", "")

        state.status = "completed"
        return {
            "state": state,
            "profile_path": profile_path,
            "bronze_sttm": bronze_sttm,
            "bronze_sttm_path": bronze_sttm_path,
            "bronze_outputs": bronze_outputs,
            "silver_sttm": silver_sttm,
            "silver_sttm_path": silver_sttm_path,
            "silver_outputs": silver_outputs,
            "gold_sttm": gold_sttm,
            "gold_sttm_path": gold_sttm_path,
            "gold_outputs": gold_outputs,
            "report": report,
        }

    except Exception as exc:
        state.error = str(exc)
        state.status = "failed"
        raise
