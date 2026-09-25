from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from core.config import (
    LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL,
    OPENAI_API_KEY, OPENAI_MODEL,
    GOOGLE_API_KEY, GEMINI_MODEL,
    GITHUB_TOKEN, GITHUB_MODEL, GITHUB_BASE_URL,
)


def _make_llm() -> Any:
    provider = (LLM_PROVIDER or "").strip().lower()
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, reasoning_format="hidden", reasoning_effort="low")
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model=OPENAI_MODEL)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model=GEMINI_MODEL)
    if provider == "github":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=GITHUB_TOKEN, model=GITHUB_MODEL, base_url=GITHUB_BASE_URL)
    return None


def _parse_llm_json(content: str) -> dict:
    content = content.strip()
    if content.startswith("```"):
        parts = content.split("```")
        content = parts[1] if len(parts) > 1 else content
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        return {}


def _default_transform_for_col(col: str, dtype: str) -> str:
    """Heuristic default transformation rules by column name + dtype."""
    col_l = col.lower()
    if any(k in col_l for k in ["amount", "price", "revenue", "total", "profit"]):
        return "cast_float"
    if any(k in col_l for k in ["quantity", "qty", "count", "units"]):
        return "cast_int"
    if any(k in col_l for k in ["date", "time", "timestamp"]):
        return "parse_date"
    if any(k in col_l for k in ["_id", "code", "sku", "ref"]):
        return "upper"
    if any(k in col_l for k in ["name", "category", "region", "city", "state",
                                  "segment", "method", "product", "store"]):
        return "title_case"
    if "email" in col_l or "url" in col_l:
        return "lower"
    if "int" in dtype or "float" in dtype:
        return "keep_numeric"
    return "strip"


def _apply_transform(series: pd.Series, transform: str) -> pd.Series:
    """Apply a named transformation to a pandas Series."""
    try:
        if transform == "cast_float":
            return pd.to_numeric(series, errors="coerce").astype("float64")
        if transform == "cast_int":
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if transform == "parse_date":
            return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        if transform == "title_case":
            return series.astype(str).str.strip().str.title().replace("Nan", pd.NA)
        if transform == "upper":
            return series.astype(str).str.strip().str.upper().replace("NAN", pd.NA)
        if transform == "lower":
            return series.astype(str).str.strip().str.lower().replace("nan", pd.NA)
        if transform == "strip":
            return series.astype(str).str.strip().replace("nan", pd.NA)
        if transform == "keep_numeric":
            return pd.to_numeric(series, errors="coerce")
    except Exception:
        pass
    return series


def _get_llm_transform_plan(df: pd.DataFrame, business_intent: str, llm: Any) -> dict[str, str]:
    """Ask LLM to return a column → transformation dict for this DataFrame."""
    data_cols = [c for c in df.columns if not c.startswith("_")]
    col_info = {c: str(df[c].dtype) for c in data_cols}
    null_counts = {c: int(df[c].isnull().sum()) for c in data_cols}
    sample = df[data_cols].head(3).to_dict(orient="records")

    prompt = f"""You are a data engineer designing Silver-layer cleaning transformations.

Business intent: "{business_intent}"
Columns and dtypes: {json.dumps(col_info)}
Null counts: {json.dumps(null_counts)}
Sample rows (3): {json.dumps(sample, default=str)[:1500]}

For each column, return the single best transformation from this list:
  cast_float   → convert to decimal number (for monetary/numeric columns)
  cast_int     → convert to integer (for quantity/count columns)
  parse_date   → parse to ISO datetime (for date/time columns)
  title_case   → title-case string (for name/category/region columns)
  upper        → uppercase string (for ID/code columns)
  lower        → lowercase string (for email/url columns)
  strip        → strip whitespace only
  keep_numeric → leave numeric columns as-is

Return ONLY valid JSON: {{"column_name": "transformation", ...}}
Include every non-metadata column. Do not include columns starting with "_"."""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        plan = _parse_llm_json(content)
        if plan and all(isinstance(v, str) for v in plan.values()):
            return plan
    except Exception:
        pass
    return {}


def silver_clean(
    bronze_output_paths: Sequence[str | Path],
    output_dir: str | Path = "data/silver_layer",
    business_intent: str = "",
) -> list[str]:
    """Clean and standardise Bronze Parquet data with LLM-driven transformations.

    Reads from Bronze Parquet outputs. For each file:
    1. Asks the LLM for a per-column transformation plan aligned to business intent.
    2. Applies transformations: type casting, date parsing, title/upper/lower casing.
    3. Deduplicates, drops all-null rows, fills nulls in key columns.
    4. Strips whitespace from all remaining string columns.
    5. Writes cleaned Parquet to the Silver layer.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    llm = _make_llm()
    written: list[str] = []

    for path in bronze_output_paths:
        src = Path(path)
        if not src.exists():
            continue

        df = pd.read_parquet(src)
        data_cols = [c for c in df.columns if not c.startswith("_")]

        # ── Step 1: Get LLM transformation plan ─────────────────────────────
        if llm and business_intent:
            llm_plan = _get_llm_transform_plan(df[data_cols], business_intent, llm)
        else:
            llm_plan = {}

        # Build final transform map: LLM overrides heuristic defaults
        transform_map: dict[str, str] = {}
        for col in data_cols:
            heuristic = _default_transform_for_col(col, str(df[col].dtype))
            transform_map[col] = llm_plan.get(col, heuristic)

        # ── Step 2: Apply column-level transformations ───────────────────────
        applied_log: list[str] = []
        for col, transform in transform_map.items():
            if col not in df.columns:
                continue
            before_nulls = df[col].isnull().sum()
            df[col] = _apply_transform(df[col], transform)
            after_nulls = df[col].isnull().sum()
            applied_log.append(f"{col}:{transform}(nulls {before_nulls}→{after_nulls})")

        # ── Step 3: Strip whitespace from any remaining object columns ───────
        for col in df.select_dtypes(include=["object"]).columns:
            if col.startswith("_"):
                continue
            try:
                df[col] = df[col].str.strip()
            except Exception:
                pass

        # ── Step 4: Deduplication ────────────────────────────────────────────
        before_rows = len(df)
        df = df.drop_duplicates()
        df = df.dropna(how="all")
        after_rows = len(df)
        dropped = before_rows - after_rows

        # ── Step 5: Null handling for key join columns ───────────────────────
        for key_col in ["product_id", "store_id"]:
            if key_col in df.columns:
                null_mask = df[key_col].isnull() | (df[key_col].astype(str).str.strip() == "")
                if null_mask.any():
                    df.loc[null_mask, key_col] = "UNKNOWN"

        # ── Step 6: Silver metadata ──────────────────────────────────────────
        df["_silver_cleaned_at"]      = pd.Timestamp.now().isoformat()
        df["_silver_business_intent"] = business_intent or "general_analytics"
        df["_silver_rows_dropped"]    = dropped
        df["_silver_transforms"]      = "; ".join(applied_log[:20])
        df["_silver_llm_guided"]      = bool(llm_plan)

        target = output / f"{src.stem.replace('_bronze', '')}_silver.parquet"
        df.to_parquet(target, index=False)
        written.append(str(target))

    return written
