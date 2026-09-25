from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bronze import bronze_ingest
from agents.gold import gold_aggregate
from agents.reporter import create_report_insight
from agents.silver import silver_clean

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
DEFAULT_FILES = [
    LANDING_DIR / "sales_data.csv",
    LANDING_DIR / "products.csv",
    LANDING_DIR / "stores.csv",
]


def refresh_landing_files() -> list[str]:
    return sorted(path.name for path in LANDING_DIR.glob("*.csv"))


def summarize_files(file_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in file_paths:
        p = Path(path)
        try:
            df = pd.read_csv(p)
            rows.append({
                "file": p.name,
                "rows": len(df),
                "columns": len(df.columns),
                "status": "ready",
            })
        except Exception:
            rows.append({
                "file": p.name,
                "rows": 0,
                "columns": 0,
                "status": "not-ready",
            })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Retail Revenue Intelligence", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 0.8rem; }
    .topbar { background: #0a2034; color: white; padding: 16px 18px; border-radius: 12px; margin-bottom: 14px; }
    .report-card { background: #eef8f4; padding: 14px; border-left: 4px solid #00a89b; border-radius: 10px; }
    .business-question { background: #ffffff; border: 1px solid #d8e0e7; border-radius: 10px; padding: 12px; }
    .layer-status { background: #eefaf9; border-radius: 8px; padding: 10px; margin: 8px 0; }
    .muted { color: #65758a; font-size: 11px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='topbar'><h1>Retail Revenue Intelligence</h1><p>Upload files and generate a simple business report.</p></div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload")
    uploaded_files = st.file_uploader("Choose CSV files", type=["csv"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded in uploaded_files:
            destination = LANDING_DIR / uploaded.name
            destination.write_bytes(uploaded.getvalue())
        st.success(f"Successfully loaded {len(uploaded_files)} file(s)")
        st.toast("Files successfully loaded into the landing area.", icon="✅")

    st.header("Business Question")
    business_question = st.text_area(
        "Ask a business question",
        value="What is the yearly and monthly revenue by product and location for the last 3 years?",
        height=120,
    )

    st.header("Workflow")
    if st.button("Run Workflow"):
        default_sources = [str(LANDING_DIR / name) for name in ["sales_data.csv", "products.csv", "stores.csv"]]
        selected_files = default_sources
        if uploaded_files:
            selected_files = [str(LANDING_DIR / uploaded.name) for uploaded in uploaded_files]

        if not selected_files:
            st.warning("Please upload at least one source CSV file.")
        else:
            file_paths = selected_files
            st.session_state["selected_files"] = file_paths
            st.session_state["business_question"] = business_question

            try:
                st.info("Bronze layer running...")
                bronze_outputs = bronze_ingest(file_paths, output_dir=PROJECT_ROOT / "data" / "bronze_layer")
                st.toast("Bronze layer completed.", icon="✅")

                st.info("Silver layer running...")
                silver_outputs = silver_clean(bronze_outputs, output_dir=PROJECT_ROOT / "data" / "silver_layer", business_intent=business_question)
                st.toast("Silver layer completed.", icon="✅")

                st.info("Gold layer running...")
                gold_outputs = gold_aggregate(silver_outputs, output_dir=PROJECT_ROOT / "data" / "gold_layer", business_intent=business_question)
                st.toast("Gold layer completed.", icon="✅")

                st.info("Reporter running...")
                report = create_report_insight(
                    gold_outputs if gold_outputs else [str(PROJECT_ROOT / "data" / "gold_layer")],
                    report_path=PROJECT_ROOT / "reports",
                    business_question=business_question,
                    source_files=file_paths,
                )
                st.toast("Reporter completed. Business report generated.", icon="✅")

                st.session_state["workflow_complete"] = True
                st.session_state["report"] = report
                st.session_state["selected_files"] = file_paths
                st.success("Workflow completed successfully.")

            except Exception as exc:
                st.error(f"Workflow failed: {exc}")

if "report" in st.session_state:
    report = st.session_state["report"]
    analysis = report.get("analysis", {})

    col1, col2, col3 = st.columns(3)
    if analysis.get("yearly_revenue"):
        latest_year = max(row["year"] for row in analysis["yearly_revenue"])
        revenue_total = sum(row.get("revenue", 0) for row in analysis["yearly_revenue"])
        col1.metric("Latest Year", str(latest_year))
        col2.metric("Revenue Total", f"${revenue_total:,.0f}")
        col3.metric("Analysis Scope", "3 Years")

    st.subheader("Business Question")
    st.markdown(f"<div class='business-question'>{report.get('business_question')}</div>", unsafe_allow_html=True)

    st.subheader("Executive Summary")
    for insight in report.get("insights", []):
        st.markdown(f"<div class='report-card'>• {insight}</div>", unsafe_allow_html=True)

    if analysis.get("yearly_revenue"):
        st.subheader("Yearly Revenue")
        yearly_revenue_df = pd.DataFrame(analysis["yearly_revenue"])
        st.bar_chart(yearly_revenue_df.set_index("year")["revenue"], width="stretch")

    if analysis.get("yearly_profit"):
        st.subheader("Yearly Profit")
        yearly_profit_df = pd.DataFrame(analysis["yearly_profit"])
        st.bar_chart(yearly_profit_df.set_index("year")["profit"], width="stretch")

    if analysis.get("monthly_revenue"):
        st.subheader("Monthly Revenue")
        monthly_revenue_df = pd.DataFrame(analysis["monthly_revenue"])
        monthly_revenue_df["period"] = monthly_revenue_df["year"].astype(str) + "-" + monthly_revenue_df["month_name"]
        st.line_chart(monthly_revenue_df.set_index("period")["revenue"], width="stretch")

    if analysis.get("top_products_by_total_sales"):
        st.subheader("Top Products")
        top_products_df = pd.DataFrame(analysis["top_products_by_total_sales"])
        st.dataframe(top_products_df, width="stretch")

    st.subheader("Download")
    st.download_button(
        label="Download the report JSON",
        data=json.dumps(report, indent=2),
        file_name="retail_report.json",
        mime="application/json",
    )

else:
    st.subheader("Upload and Ask a Business Question")
    st.write("Select the sales, products, and stores CSV files, then run the workflow to create a report.")

    selected_preview = []
    for path in [str(LANDING_DIR / "sales_data.csv"), str(LANDING_DIR / "products.csv"), str(LANDING_DIR / "stores.csv")]:
        if Path(path).exists():
            selected_preview.append(path)
    if selected_preview:
        preview_df = summarize_files(selected_preview)
        st.dataframe(preview_df, width="stretch")
