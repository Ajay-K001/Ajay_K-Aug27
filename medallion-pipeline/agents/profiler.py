from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from core.config import (
    LLM_PROVIDER,
    PROFILES_DIR,
    GROQ_API_KEY,
    GROQ_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GOOGLE_API_KEY,
    GEMINI_MODEL,
    GITHUB_TOKEN,
    GITHUB_MODEL,
    GITHUB_BASE_URL,
)
from core.observability import AgentTrace


PROFILER_AGENT_PROMPT = """
You are a Data Analyst specializing in data profiling.
Follow this sequence: THINK → INSPECT → PLAN → ACT → VERIFY.
First call inspect_files_tool to preview files.
Then call profiler_tool to get full statistics.
Return JSON with: semantic_meanings, join_keys, quality_notes.
""".strip()


def _normalize_file_paths(file_paths: Sequence[str | Path]) -> list[Path]:
    """Return file Paths and validate they exist."""
    paths = [Path(p) for p in file_paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        if path.suffix.lower() != ".csv":
            raise ValueError(f"Only CSV files are supported: {path}")
    return paths


def _inspect_files(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Read each CSV and return shape, columns, dtypes, and three sample values per column.

    This is a pure-Python helper with no LLM dependency.
    """
    file_paths = _normalize_file_paths(file_paths)
    result: dict[str, Any] = {}

    for path in file_paths:
        df = pd.read_csv(path)
        sample_values: dict[str, list[Any]] = {}

        for column in df.columns:
            series = df[column].dropna()
            sample = []
            for value in series.head(3).tolist():
                sample.append(value)
            sample_values[column] = sample

        result[path.name] = {
            "path": str(path),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "dtypes": {column: str(df[column].dtype) for column in df.columns},
            "sample_values": sample_values,
        }

    return result


def _compute_stats(file_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Compute CSV-level profiling statistics for all files.

    Returns dtype, null_count, null_pct, unique_count, numeric min/max/mean,
    and sample text values for columns.
    """
    file_paths = _normalize_file_paths(file_paths)
    result: dict[str, Any] = {}

    for path in file_paths:
        df = pd.read_csv(path)
        column_stats: dict[str, Any] = {}

        for column in df.columns:
            series = df[column]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = float((series.isna().mean()) * 100) if len(series) else 0.0
            unique_count = int(series.nunique(dropna=True))

            stat: dict[str, Any] = {
                "dtype": dtype,
                "null_count": null_count,
                "null_pct": round(null_pct, 4),
                "unique_count": unique_count,
            }

            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() >= 1:
                non_null_numeric = numeric_series.dropna()
                stat["min"] = float(non_null_numeric.min())
                stat["max"] = float(non_null_numeric.max())
                stat["mean"] = float(non_null_numeric.mean())

            text_sample = []
            for value in series.dropna().astype(str).head(3).tolist():
                text_sample.append(value)
            stat["sample_values"] = text_sample

            column_stats[column] = stat

        result[path.name] = {
            "path": str(path),
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "columns": list(df.columns),
            "stats": column_stats,
        }

    return result


def _make_profiler_tools(file_paths: Sequence[str | Path], run_id: str) -> dict[str, Any]:
    """Create LangChain tool callables for file inspection and profiling.

    The returned dict contains:
      - inspect_files_tool
      - profiler_tool
    """
    normalized_paths = _normalize_file_paths(file_paths)

    @tool
    def inspect_files_tool() -> str:
        """Preview CSV shape, columns, dtypes, and up to three sample values per column."""
        return json.dumps(_inspect_files(normalized_paths))

    @tool
    def profiler_tool() -> str:
        """Compute full statistics for all CSV files in the requested dataset list."""
        return json.dumps(_compute_stats(normalized_paths))

    return {
        "inspect_files_tool": inspect_files_tool,
        "profiler_tool": profiler_tool,
    }


def _make_llm() -> Any:
    """Create the configured LLM client for the profiler agent.

    Uses ChatGroq, ChatOpenAI, or ChatGoogleGenerativeAI according to LLM_PROVIDER.
    For Groq, hide hidden reasoning tags by setting reasoning_format="hidden"
    and reasoning_effort="low".
    """
    provider = (LLM_PROVIDER or "").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            reasoning_format="hidden",
            reasoning_effort="low",
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)

    if provider == "github":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=GITHUB_TOKEN,
            model=GITHUB_MODEL,
            base_url=GITHUB_BASE_URL,
        )

    raise ValueError(
        "LLM_PROVIDER is not configured. Set one of: groq, openai, gemini, github."
    )


def _create_agent(file_paths: Sequence[str | Path], run_id: str) -> Any:
    """Create the LangGraph React agent from the profiler tools and LLM."""
    tools = _make_profiler_tools(file_paths, run_id)
    llm = _make_llm()
    agent = create_react_agent(
        llm,
        tools=list(tools.values()),
        prompt=PROFILER_AGENT_PROMPT,
    )
    return agent


def profile_dataset(file_path: str | Path, run_id: str, task_description: str) -> str:
    """Profile a single CSV file and write the combined profile JSON to PROFILES_DIR.

    Returns the profile file path as a string.
    """
    return profile_multiple_datasets([file_path], run_id, task_description)


def profile_multiple_datasets(
    file_paths: Sequence[str | Path], run_id: str, task_description: str
) -> str:
    """Profile one or more CSV files with the profiler agent and return the JSON file path.

    The profile is saved as PROFILES_DIR / f"profile_combined_{timestamp}.json".
    """
    file_paths = _normalize_file_paths(file_paths)
    trace = AgentTrace("profiler", run_id).set_input(
        task_description=task_description,
        file_paths=[str(path) for path in file_paths],
    )
    trace.set_plan(
        "THINK → INSPECT → PLAN → ACT → VERIFY"
    )

    try:
        agent = _create_agent(file_paths, run_id)
        # Run the React agent using the requested task description.
        # This catches the inspect and profiler tools in the required order.
        result = agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        f"{task_description}\n\n"
                        f"Inspect these CSV files and profile them. "
                        f"Return JSON with semantic_meanings, join_keys, and quality_notes.",
                    )
                ]
            }
        )

        # Convert the tool outputs into a combined dictionary.
        inspect_data = _inspect_files(file_paths)
        stats_data = _compute_stats(file_paths)

        combined_profile: dict[str, Any] = {
            "run_id": run_id,
            "task_description": task_description,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "semantic_meanings": [],
            "join_keys": [],
            "quality_notes": [],
            "inspect": inspect_data,
            "stats": stats_data,
            "agent_result": result,
        }

        # Save under the configured profile directory.
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        profile_path = PROFILES_DIR / f"profile_combined_{timestamp}.json"
        with profile_path.open("w", encoding="utf-8") as file:
            json.dump(combined_profile, file, indent=2, default=str)

        trace.set_output(profile_path=str(profile_path))
        trace.complete(status="success")
        return str(profile_path)

    except Exception as exc:
        trace.fail(str(exc))
        raise
