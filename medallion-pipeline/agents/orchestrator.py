from __future__ import annotations

from pathlib import Path
from typing import Any

from core.state import PipelineState


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
