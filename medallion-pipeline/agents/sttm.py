from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence


def generate_bronze_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Create a Bronze STTM mapping template from the input file list."""
    files = [str(Path(p)) for p in file_paths]
    return {
        "phase": "bronze",
        "source_to_target": {
            "source_files": files,
            "target_layer": "bronze_layer",
            "mapping": [
                {"source": "raw_csv", "target": "bronze_parquet", "operation": "ingest"}
            ],
        },
    }


def generate_silver_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Create a Silver STTM mapping template from the input file list."""
    files = [str(Path(p)) for p in file_paths]
    return {
        "phase": "silver",
        "source_to_target": {
            "source_files": files,
            "target_layer": "silver_layer",
            "mapping": [
                {"source": "bronze", "target": "silver", "operation": "clean"}
            ],
        },
    }


def generate_gold_sttm(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Create a Gold STTM mapping template from the input file list."""
    files = [str(Path(p)) for p in file_paths]
    return {
        "phase": "gold",
        "source_to_target": {
            "source_files": files,
            "target_layer": "gold_layer",
            "mapping": [
                {"source": "silver", "target": "gold", "operation": "aggregate"}
            ],
        },
    }
