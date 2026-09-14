import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_profiler_agent_module_exists_and_exports_public_api():
    module = importlib.import_module("agents.profiler")
    assert hasattr(module, "PROFILER_AGENT_PROMPT")
    assert hasattr(module, "_inspect_files")
    assert hasattr(module, "_compute_stats")
    assert hasattr(module, "_make_profiler_tools")
    assert hasattr(module, "_make_llm")
    assert hasattr(module, "profile_dataset")
    assert hasattr(module, "profile_multiple_datasets")


def test_required_pipeline_agent_files_exist():
    required = [
        PROJECT_ROOT / "agents" / "profiler.py",
        PROJECT_ROOT / "agents" / "sttm.py",
        PROJECT_ROOT / "agents" / "bronze.py",
        PROJECT_ROOT / "agents" / "silver.py",
        PROJECT_ROOT / "agents" / "gold.py",
        PROJECT_ROOT / "agents" / "reporter.py",
        PROJECT_ROOT / "agents" / "orchestrator.py",
    ]
    for path in required:
        assert path.exists(), f"Missing required agent file: {path}"
