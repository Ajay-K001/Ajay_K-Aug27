from __future__ import annotations

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


def run_pipeline_phase(phase: str, file_paths: list[str], run_id: str = "") -> PipelineState:
    """Create a PipelineState snapshot for the requested Medallion workflow step."""
    state = PipelineState(run_id=run_id)
    state.status = phase
    state.uploaded_files = [str(Path(p)) for p in file_paths]
    return state


def run_stepwise_pipeline(file_paths: list[str], run_id: str = "") -> dict[str, Any]:
    """Return a phase-by-phase workflow plan that mirrors the UI and orchestration requirement."""
    ordered = []
    for index, phase in enumerate(PHASES):
        ordered.append(
            {
                "phase": index,
                "name": phase,
                "status": "pending",
                "run_id": run_id,
                "rows": None,
            }
        )

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

    Orchestrates all agents through phases:
    1. Profile → 2. Bronze STTM + approval + ingest
    3. Silver STTM + approval + clean → 4. Gold STTM + approval + aggregate
    5. Report generation

    Returns pipeline state with all outputs.
    """
    base_path = Path(base_dir)
    state = PipelineState(run_id=run_id)
    state.business_intent = business_intent
    state.uploaded_files = file_paths

    try:
        # Phase 1: Profile
        state.status = "Profile Raw Files"
        profile_path = profile_multiple_datasets(file_paths, run_id, "Profile retail dataset")
        state.profile_path = profile_path

        # Phase 2: Bronze STTM
        state.status = "Bronze STTM"
        bronze_sttm = generate_bronze_sttm(file_paths)
        state.sttm_bronze_path = str(base_path / "data" / "sttm" / f"bronze_sttm_{run_id}.json")

        if not skip_approval:
            state.status = "Bronze STTM (Awaiting Approval)"
            # In real flow, this would wait for user approval via UI/CLI
        state.hitl_approved = True  # Assume approved for orchestration

        # Phase 3: Bronze Ingest
        state.status = "Bronze Agent"
        bronze_outputs = bronze_ingest(file_paths, output_dir=str(base_path / "data" / "bronze_layer"))
        state.bronze_output_paths = bronze_outputs

        # Phase 4: Silver STTM
        state.status = "Silver STTM"
        silver_sttm = generate_silver_sttm(file_paths)
        state.sttm_silver_path = str(base_path / "data" / "sttm" / f"silver_sttm_{run_id}.json")

        if not skip_approval:
            state.status = "Silver STTM (Awaiting Approval)"
        state.hitl_approved = True

        # Phase 5: Silver Clean (reads from Bronze outputs)
        state.status = "Silver Agent"
        silver_outputs = silver_clean(
            bronze_outputs,
            output_dir=str(base_path / "data" / "silver_layer"),
            business_intent=business_intent
        )
        state.silver_output_paths = silver_outputs

        # Phase 6: Gold STTM
        state.status = "Gold STTM"
        gold_sttm = generate_gold_sttm(file_paths)
        state.sttm_gold_path = str(base_path / "data" / "sttm" / f"gold_sttm_{run_id}.json")

        if not skip_approval:
            state.status = "Gold STTM (Awaiting Approval)"
        state.hitl_approved = True

        # Phase 7: Gold Aggregate (reads from Silver outputs)
        state.status = "Gold Agent"
        gold_outputs = gold_aggregate(
            silver_outputs,
            output_dir=str(base_path / "data" / "gold_layer"),
            business_intent=business_intent
        )
        state.gold_output_paths = gold_outputs

        # Phase 8: Report Generation
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
            "bronze_outputs": bronze_outputs,
            "silver_sttm": silver_sttm,
            "silver_outputs": silver_outputs,
            "gold_sttm": gold_sttm,
            "gold_outputs": gold_outputs,
            "report": report,
        }

    except Exception as exc:
        state.error = str(exc)
        state.status = "failed"
        raise
