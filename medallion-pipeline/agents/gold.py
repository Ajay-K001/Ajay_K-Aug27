from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from core.config import (
    LLM_PROVIDER,
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


def _identify_role(df: pd.DataFrame) -> str:
    cols = set(c.lower() for c in df.columns)
    if "transaction_date" in cols or "total_amount" in cols:
        return "sales"
    if "standard_price" in cols or ("product_name" in cols and "product_id" in cols and "transaction_date" not in cols):
        return "products"
    if "store_name" in cols or "region" in cols:
        return "stores"
    return "unknown"


def _safe_parquet_path(path: str) -> str:
    return path.replace("\\", "/")


def gold_aggregate(
    silver_output_paths: Sequence[str | Path],
    output_dir: str | Path = "data/gold_layer",
    business_intent: str = "",
) -> list[str]:
    """Merge Silver Parquets with DuckDB JOIN and materialise Gold KPI tables.

    Joins sales + products + stores on their _id columns, then writes one
    Parquet per KPI target: revenue_by_product, revenue_by_region,
    revenue_by_period, and fact_sales_gold (full enriched fact table).
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Classify Silver Parquets by content
    role_map: dict[str, pd.DataFrame] = {}
    for path in silver_output_paths:
        p = Path(path)
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
            role = _identify_role(df)
            if role not in role_map:
                role_map[role] = df
        except Exception:
            pass

    if "sales" not in role_map:
        return []

    con = duckdb.connect()
    con.register("sales", role_map["sales"])

    # Get LLM guidance on KPIs
    llm = _make_llm()
    aggregation_guidance = ""
    if llm and business_intent:
        try:
            sales_cols = [c for c in role_map["sales"].columns if not c.startswith("_")]
            prompt = (
                f'Business question: "{business_intent}"\n'
                f"Sales columns: {sales_cols}\n"
                "Describe the 3 most important KPI aggregations in 2 sentences."
            )
            response = llm.invoke(prompt)
            aggregation_guidance = (response.content[:400] if hasattr(response, "content") else "")
        except Exception:
            pass

    # Build fact table: sales LEFT JOIN products LEFT JOIN stores
    select_parts = ["s.*"]
    from_clause = "FROM sales s"

    if "products" in role_map:
        con.register("products", role_map["products"])
        prod_cols = role_map["products"].columns.tolist()
        extras = [c for c in ["product_name", "category", "standard_price"]
                  if c in prod_cols and c not in role_map["sales"].columns]
        if extras and "product_id" in prod_cols:
            select_parts += [f"p.{c}" for c in extras]
            from_clause += " LEFT JOIN products p ON s.product_id = p.product_id"

    if "stores" in role_map:
        con.register("stores", role_map["stores"])
        store_cols = role_map["stores"].columns.tolist()
        extras = [c for c in ["store_name", "region", "city", "state"]
                  if c in store_cols and c not in role_map["sales"].columns]
        if extras and "store_id" in store_cols:
            select_parts += [f"st.{c}" for c in extras]
            from_clause += " LEFT JOIN stores st ON s.store_id = st.store_id"

    con.execute(
        f"CREATE OR REPLACE TABLE fact AS SELECT {', '.join(select_parts)} {from_clause}"
    )

    # Add year/month columns for time-based analysis
    fact_cols = [row[0] for row in con.execute("DESCRIBE fact").fetchall()]
    if "transaction_date" in fact_cols:
        con.execute("""
            CREATE OR REPLACE TABLE fact AS
            SELECT *,
                   TRY_CAST(year(TRY_CAST(transaction_date AS TIMESTAMP)) AS INTEGER)  AS _year,
                   TRY_CAST(month(TRY_CAST(transaction_date AS TIMESTAMP)) AS INTEGER) AS _month,
                   monthname(TRY_CAST(transaction_date AS TIMESTAMP))                  AS _month_name
            FROM fact
        """)
        fact_cols = [row[0] for row in con.execute("DESCRIBE fact").fetchall()]

    written: list[str] = []

    # ── KPI 1: Revenue by Product ────────────────────────────────────────────
    has_product = "product_name" in fact_cols
    has_amount = "total_amount" in fact_cols
    has_year = "_year" in fact_cols
    has_region = "region" in fact_cols
    has_city = "city" in fact_cols
    has_state = "state" in fact_cols

    if has_product and has_amount:
        dim = "_year, product_name" if has_year else "product_name"
        qty_col = "SUM(quantity) as total_quantity," if "quantity" in fact_cols else ""
        df_prod = con.execute(f"""
            SELECT {dim},
                   SUM(total_amount)  AS total_revenue,
                   {qty_col}
                   COUNT(*)           AS transaction_count,
                   AVG(total_amount)  AS avg_order_value
            FROM fact
            WHERE product_name IS NOT NULL
            GROUP BY {dim}
            ORDER BY total_revenue DESC
        """).fetchdf()
        df_prod["_gold_intent"] = business_intent
        df_prod["_gold_llm_guidance"] = aggregation_guidance
        path = output / "revenue_by_product.parquet"
        df_prod.to_parquet(path, index=False)
        written.append(str(path))

    # ── KPI 2: Revenue by Region ─────────────────────────────────────────────
    if (has_region or has_city) and has_amount:
        dims = []
        if has_year:
            dims.append("_year")
        if has_region:
            dims.append("region")
        if has_city:
            dims.append("city")
        if has_state:
            dims.append("state")
        if dims:
            dim_str = ", ".join(dims)
            df_region = con.execute(f"""
                SELECT {dim_str},
                       SUM(total_amount) AS total_revenue,
                       COUNT(*)          AS transaction_count,
                       AVG(total_amount) AS avg_order_value
                FROM fact
                WHERE {dims[0]} IS NOT NULL
                GROUP BY {dim_str}
                ORDER BY total_revenue DESC
            """).fetchdf()
            df_region["_gold_intent"] = business_intent
            path = output / "revenue_by_region.parquet"
            df_region.to_parquet(path, index=False)
            written.append(str(path))

    # ── KPI 3: Revenue by Period (month/year) ────────────────────────────────
    if has_year and has_amount:
        df_period = con.execute("""
            SELECT _year, _month, _month_name,
                   SUM(total_amount) AS total_revenue,
                   COUNT(*)          AS transaction_count
            FROM fact
            WHERE _year IS NOT NULL
            GROUP BY _year, _month, _month_name
            ORDER BY _year, _month
        """).fetchdf()
        df_period["_gold_intent"] = business_intent
        path = output / "revenue_by_period.parquet"
        df_period.to_parquet(path, index=False)
        written.append(str(path))

    # ── Full enriched fact table ─────────────────────────────────────────────
    full_df = con.execute("SELECT * FROM fact").fetchdf()
    full_df["_gold_aggregated_at"] = pd.Timestamp.now().isoformat()
    full_df["_gold_intent"] = business_intent
    full_df["_gold_llm_guidance"] = aggregation_guidance
    path = output / "fact_sales_gold.parquet"
    full_df.to_parquet(path, index=False)
    written.append(str(path))

    con.close()
    return written
