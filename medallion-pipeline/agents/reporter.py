from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import plotly.express as px
import plotly.io as pio


SALES_GOLD_TABLE = "fact_sales_gold"


def _relevant_topics(business_question: str) -> set[str]:
    """Return which KPI topics are relevant to the business question."""
    q = (business_question or "").lower()
    topics: set[str] = set()
    if any(k in q for k in ["product", "item", "sku", "profitable", "best sell", "top", "rank"]):
        topics.add("product")
    if any(k in q for k in ["region", "location", "city", "state", "area", "where", "geographic", "store"]):
        topics.add("region")
    if any(k in q for k in ["month", "monthly", "trend", "seasonal", "quarter"]):
        topics.add("monthly")
    if any(k in q for k in ["year", "yearly", "annual", "period", "time", "over time", "3 year", "three year"]):
        topics.add("yearly")
    # If nothing matched, show everything
    if not topics:
        topics = {"product", "region", "monthly", "yearly"}
    return topics


def _safe_path(path: str) -> str:
    return path.replace("\\", "/")


def _load_gold_tables(gold_paths: list[str], con: duckdb.DuckDBPyConnection) -> dict[str, str]:
    """Register Gold Parquet files as DuckDB tables. Returns {table_name: parquet_path}."""
    registered: dict[str, str] = {}
    for path in gold_paths:
        p = Path(path)
        if not p.exists() or not str(p).endswith(".parquet"):
            continue
        table_name = p.stem.replace("-", "_").replace(" ", "_")
        try:
            con.execute(
                f"CREATE OR REPLACE TABLE {table_name} AS "
                f"SELECT * FROM read_parquet('{_safe_path(str(p))}')"
            )
            registered[table_name] = str(p)
        except Exception:
            pass
    return registered


def _execute_query(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    """Execute SQL against DuckDB and return row dicts. Retries once on error."""
    try:
        return con.execute(sql).fetchdf().to_dict(orient="records")
    except Exception as e:
        try:
            return con.execute(sql.replace("TRY_CAST", "CAST")).fetchdf().to_dict(orient="records")
        except Exception:
            return []


def _make_charts(con: duckdb.DuckDBPyConnection, tables: dict[str, str], business_question: str, topics: set[str] | None = None) -> list[dict]:
    """Generate Plotly charts from Gold tables. Returns list of {title, type, json} dicts."""
    if topics is None:
        topics = _relevant_topics(business_question)
    charts: list[dict] = []

    # Chart 1: Top products by revenue
    if "product" in topics and "revenue_by_product" in tables:
        try:
            df = con.execute("""
                SELECT product_name,
                       SUM(total_revenue) AS revenue
                FROM revenue_by_product
                WHERE product_name IS NOT NULL
                GROUP BY product_name
                ORDER BY revenue DESC
                LIMIT 10
            """).fetchdf()
            if not df.empty:
                fig = px.bar(
                    df, x="product_name", y="revenue",
                    title="Top 10 Products by Revenue",
                    labels={"product_name": "Product", "revenue": "Revenue ($)"},
                    color="revenue", color_continuous_scale="teal",
                )
                fig.update_layout(xaxis_tickangle=-35, showlegend=False)
                charts.append({"title": "Top 10 Products by Revenue", "type": "bar", "json": pio.to_json(fig)})
        except Exception:
            pass

    # Chart 2: Revenue by region
    if "region" in topics and "revenue_by_region" in tables:
        try:
            df = con.execute("""
                SELECT region, SUM(total_revenue) AS revenue
                FROM revenue_by_region
                WHERE region IS NOT NULL
                GROUP BY region
                ORDER BY revenue DESC
            """).fetchdf()
            if not df.empty:
                fig = px.pie(df, names="region", values="revenue", title="Revenue by Region")
                charts.append({"title": "Revenue by Region", "type": "pie", "json": pio.to_json(fig)})
        except Exception:
            pass

    # Chart 3: Monthly revenue trend
    if "monthly" in topics and "revenue_by_period" in tables:
        try:
            df = con.execute("""
                SELECT _year, _month, _month_name,
                       SUM(total_revenue) AS revenue
                FROM revenue_by_period
                WHERE _year IS NOT NULL
                GROUP BY _year, _month, _month_name
                ORDER BY _year, _month
            """).fetchdf()
            if not df.empty:
                df["period"] = df["_year"].astype(str) + "-" + df["_month_name"].fillna("")
                fig = px.line(
                    df, x="period", y="revenue",
                    title="Monthly Revenue Trend",
                    labels={"period": "Period", "revenue": "Revenue ($)"},
                    markers=True,
                )
                charts.append({"title": "Monthly Revenue Trend", "type": "line", "json": pio.to_json(fig)})
        except Exception:
            pass

    # Chart 4: Yearly revenue bar (from period table)
    if "yearly" in topics and "revenue_by_period" in tables:
        try:
            df = con.execute("""
                SELECT _year, SUM(total_revenue) AS revenue
                FROM revenue_by_period
                WHERE _year IS NOT NULL
                GROUP BY _year
                ORDER BY _year
            """).fetchdf()
            if not df.empty:
                fig = px.bar(
                    df, x="_year", y="revenue",
                    title="Yearly Revenue",
                    labels={"_year": "Year", "revenue": "Revenue ($)"},
                    color="revenue", color_continuous_scale="teal",
                )
                charts.append({"title": "Yearly Revenue", "type": "bar", "json": pio.to_json(fig)})
        except Exception:
            pass

    return charts


def _build_analysis(con: duckdb.DuckDBPyConnection, tables: dict[str, str], topics: set[str] | None = None) -> dict[str, Any]:
    """Run SQL queries to build the structured analysis dict, filtered by relevant topics."""
    if topics is None:
        topics = {"product", "region", "monthly", "yearly"}
    analysis: dict[str, Any] = {}

    if "product" in topics and "revenue_by_product" in tables:
        rows = _execute_query(con, """
            SELECT product_name,
                   SUM(total_revenue) AS total_amount
            FROM revenue_by_product
            WHERE product_name IS NOT NULL
            GROUP BY product_name
            ORDER BY total_amount DESC
        """)
        analysis["top_products_by_total_sales"] = rows

    if "product" in topics and "yearly" in topics and "revenue_by_product" in tables:
        yearly_rows = _execute_query(con, """
            SELECT _year AS year, product_name,
                   SUM(total_revenue) AS total_amount
            FROM revenue_by_product
            WHERE product_name IS NOT NULL
            GROUP BY _year, product_name
            ORDER BY year, total_amount DESC
        """)
        analysis["yearly_sales_by_product"] = yearly_rows

    if "region" in topics and "revenue_by_region" in tables:
        rows = _execute_query(con, """
            SELECT _year AS year,
                   region, city, state,
                   SUM(total_revenue) AS total_amount
            FROM revenue_by_region
            WHERE region IS NOT NULL
            GROUP BY _year, region, city, state
            ORDER BY year, total_amount DESC
        """)
        analysis["yearly_sales_by_location"] = rows

    if "yearly" in topics and "revenue_by_period" in tables:
        yearly = _execute_query(con, """
            SELECT _year AS year,
                   SUM(total_revenue) AS revenue
            FROM revenue_by_period
            WHERE _year IS NOT NULL
            GROUP BY _year
            ORDER BY year
        """)
        analysis["yearly_revenue"] = yearly

    if "monthly" in topics and "revenue_by_period" in tables:
        monthly = _execute_query(con, """
            SELECT _year AS year, _month AS month,
                   _month_name AS month_name,
                   SUM(total_revenue) AS revenue
            FROM revenue_by_period
            WHERE _year IS NOT NULL
            GROUP BY _year, _month, _month_name
            ORDER BY year, month
        """)
        analysis["monthly_revenue"] = monthly

    return analysis


def _build_insights(analysis: dict[str, Any], business_question: str) -> list[str]:
    insights: list[str] = []
    if business_question:
        insights.append(f"Business question: {business_question}")

    top = analysis.get("top_products_by_total_sales", [])
    if top:
        best = top[0]
        insights.append(
            f"Top product by revenue: {best.get('product_name', '?')} "
            f"(${best.get('total_amount', 0):,.2f})"
        )

    locations = analysis.get("yearly_sales_by_location", [])
    if locations:
        best = max(locations, key=lambda r: r.get("total_amount", 0))
        insights.append(
            f"Strongest location: {best.get('city', '?')}, {best.get('region', '?')} "
            f"(${best.get('total_amount', 0):,.2f})"
        )

    yearly = analysis.get("yearly_revenue", [])
    if yearly:
        best_year = max(yearly, key=lambda r: r.get("revenue", 0))
        insights.append(
            f"Best revenue year: {best_year.get('year')} "
            f"(${best_year.get('revenue', 0):,.2f})"
        )

    if not insights:
        insights.append("Report generated from Gold layer data.")

    return insights


def create_report_insight(
    gold_output_paths: str | Path | list[str],
    report_path: str | Path = "reports",
    business_question: str | None = None,
    source_files: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Generate an executive report using DuckDB SQL over Gold Parquet tables.

    Loads all Gold Parquets as DuckDB tables, runs SQL aggregations to answer
    the business question, generates Plotly charts, and writes JSON + HTML reports.
    """
    report_dir = Path(report_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(gold_output_paths, (str, Path)):
        gold_paths = [str(gold_output_paths)]
    else:
        gold_paths = [str(p) for p in gold_output_paths]

    con = duckdb.connect()
    tables = _load_gold_tables(gold_paths, con)

    topics = _relevant_topics(business_question or "")

    analysis: dict[str, Any] = {}
    charts: list[dict] = []
    insights: list[str] = []

    try:
        analysis = _build_analysis(con, tables, topics)
        charts = _make_charts(con, tables, business_question or "", topics)
        insights = _build_insights(analysis, business_question or "")
    except Exception as exc:
        insights = [f"Report generation encountered an issue: {exc}"]
    finally:
        con.close()

    result: dict[str, Any] = {
        "source": gold_paths[0] if gold_paths else "",
        "summary": "Gold layer analysed via DuckDB SQL. Charts generated with Plotly.",
        "business_question": business_question or "General revenue and product analysis",
        "insights": insights,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis": analysis,
        "charts": charts,
    }

    report_file = report_dir / "report_insights.json"
    report_file.write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    _write_html(result, report_dir)
    _write_pdf(result, report_dir)

    return result


def _write_html(result: dict[str, Any], report_dir: Path) -> None:
    from html import escape

    insights_html = "\n".join(
        f"<li>{escape(str(i))}</li>" for i in result.get("insights", [])
    )

    chart_html = ""
    for chart in result.get("charts", []):
        chart_html += f"""
        <h2>{escape(chart.get('title', ''))}</h2>
        <div id="chart_{hash(chart.get('title',''))}"></div>
        <script>
          Plotly.react("chart_{hash(chart.get('title',''))}", {chart.get('json','{}')});
        </script>
        """

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Medallion Pipeline Executive Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 40px; color: #1f2937; background: #f8fafc; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 12px; padding: 28px; background: white; max-width: 960px; margin: 0 auto; }}
    h1 {{ color: #0f766e; }}
    h2 {{ color: #334155; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; }}
    ul {{ padding-left: 20px; line-height: 1.8; }}
    .meta {{ color: #64748b; font-size: 12px; margin-top: 6px; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Medallion Pipeline Executive Report</h1>
    <p class="meta">Generated: {escape(result['generated_at'])}</p>
    <h2>Business Question</h2>
    <p>{escape(str(result['business_question']))}</p>
    <h2>Key Insights</h2>
    <ul>{insights_html}</ul>
    {chart_html}
  </div>
</body>
</html>"""

    (report_dir / "report_insights.html").write_text(html, encoding="utf-8")


def _write_pdf(result: dict[str, Any], report_dir: Path) -> None:
    pdf_file = report_dir / "report_insights.pdf"
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt

        question = str(result.get("business_question", ""))
        top_products = result.get("analysis", {}).get("top_products_by_total_sales", [])

        with PdfPages(str(pdf_file)) as pdf:
            # Cover page
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis("off")
            ax.text(0.05, 0.88, "Medallion Pipeline Executive Report", fontsize=18, weight="bold", color="#0f766e")
            ax.text(0.05, 0.74, f"Business Question: {question}", fontsize=11, wrap=True)
            ax.text(0.05, 0.58, f"Generated: {result.get('generated_at', '')}", fontsize=9, color="#64748b")
            ax.text(0.05, 0.46, "Key Insights:", fontsize=12, weight="bold")
            y = 0.36
            for insight in result.get("insights", [])[:5]:
                ax.text(0.07, y, f"• {str(insight)[:110]}", fontsize=9)
                y -= 0.10
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

            # Top Products bar chart
            if top_products:
                names = [r.get("product_name", "?")[:20] for r in top_products[:10]]
                values = [float(r.get("total_amount", 0)) for r in top_products[:10]]
                fig, ax = plt.subplots(figsize=(10, 5))
                bars = ax.barh(names[::-1], values[::-1], color="#0d9488")
                ax.set_xlabel("Revenue ($)")
                ax.set_title("Top Products by Revenue")
                for bar, val in zip(bars, values[::-1]):
                    ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                            f"${val:,.0f}", va="center", fontsize=8)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

            # Yearly revenue bar chart
            yearly = result.get("analysis", {}).get("yearly_revenue", [])
            if yearly:
                years = [str(r.get("year", "")) for r in yearly]
                revenues = [float(r.get("revenue", 0)) for r in yearly]
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.bar(years, revenues, color="#0d9488")
                ax.set_xlabel("Year")
                ax.set_ylabel("Revenue ($)")
                ax.set_title("Yearly Revenue")
                for i, (y, r) in enumerate(zip(years, revenues)):
                    ax.text(i, r * 1.01, f"${r:,.0f}", ha="center", fontsize=9)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    except Exception:
        pdf_file.write_bytes(b"%PDF-1.4\n")
