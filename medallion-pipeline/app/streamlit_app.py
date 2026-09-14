from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bronze import bronze_ingest
from agents.gold import gold_aggregate
from agents.orchestrator import PHASES, run_pipeline_phase, run_stepwise_pipeline
from agents.profiler import profile_multiple_datasets
from agents.reporter import create_report_insight
from agents.silver import silver_clean
from agents.sttm import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
DEFAULT_FILES = [
    LANDING_DIR / "sales_data.csv",
    LANDING_DIR / "products.csv",
    LANDING_DIR / "stores.csv",
]


def render_workflow_status_html(workflow_status: list[dict[str, str]]) -> str:
    """Return HTML rows for the three-layer workflow lifecycle."""
    rows = []
    for item in workflow_status:
        layer = item["layer"]
        state = item["state"]
        if state == "in_progress":
            label = f"{layer} in progress"
            cls = "pending"
            dot = "status-dot pending"
        elif state == "completed":
            label = f"{layer} completed"
            cls = "complete"
            dot = "status-dot"
        else:
            label = layer
            cls = "pending"
            dot = "status-dot pending"

        rows.append(
            f"<div class='status-chip {cls}'><span class='{dot}'></span>{label}</div>"
        )
    return "<div class='workflow-status'><span class='subtle'>Workflow Status</span>" + "".join(rows) + "</div>"


def refresh_landing_files() -> list[str]:
    """Return all CSV names currently available in the landing folder, including uploaded files."""
    return sorted([path.name for path in LANDING_DIR.glob("*.csv")])


def summarize_files(file_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in file_paths:
        p = Path(path)
        df = pd.read_csv(p)
        rows.append({
            "file": p.name,
            "rows": len(df),
            "columns": len(df.columns),
            "status": "ready",
        })
    return pd.DataFrame(rows)


st.set_page_config(page_title="Medallion Pipeline", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; background: linear-gradient(135deg, #eef8f6 0%, #f8fafb 100%); max-width: 1280px; margin: 0 auto; }
    .title-panel { background: linear-gradient(135deg, #102c2f 0%, #204c54 100%); color: #eef8f8; padding: 26px 30px; border-radius: 14px; box-shadow: 0 8px 22px rgba(16,40,47,0.2); margin-bottom: 14px; }
    .title-panel h1 { color: #eef8f8; font-size: clamp(2.2rem, 3vw, 3rem); margin: 0 0 12px; }
    .title-panel p { color: #d7f5ed; font-size: 1.1rem; margin: 0; }
    .ui-grid { display: flex; gap: 24px; align-items: flex-start; }
    .left-panel { width: min(280px, 24%); background: #eef7f7; border: 1px solid #bdd7d6; border-radius: 12px; padding: 18px 16px; box-shadow: 0 8px 20px rgba(0,0,0,0.04); }
    .center-panel { flex: 1; background: #ffffff; border: 1px solid #d8e7e9; border-radius: 14px; padding: 24px; box-shadow: 0 12px 30px rgba(0,0,0,0.08); }
    .control-title { font-size: 17px; font-weight: 700; color: #20333a; margin-bottom: 10px; }
    .left-panel .stTextInput input, .left-panel .stFileUploader > div { background: #fffdfa; border-radius: 10px; border: 1px solid #b9d1d1; color: #204247; }
    .question-card { background: #ffffff; border-radius: 12px; border: 1px solid #ccdce2; padding: 16px 18px; margin: 14px 0; }
    .question-card .stTextArea textarea { background: #eef3f4; border-radius: 12px; border: 1px solid #bcd1d6; color: #20333a; }
    .suggested { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; }
    .stButton button { background: #eef5ee; color: #203d3c; border-radius: 12px; border: 1px solid #b9c9cb; padding: 10px 14px; font-weight: 700; }
    .run-button { width: 100%; background: #0a716b; color: white; border-radius: 10px; font-weight: 800; padding: 12px; }
    .workflow-status { margin-top: 14px; padding: 12px; border-radius: 12px; border: 1px solid #bfd8d8; background: #eefbfa; }
    .status-chip { display: flex; align-items: center; gap: 9px; padding: 7px 11px; margin-top: 8px; border-radius: 10px; background: #eaf7f5; color: #183d3d; font-size: 12px; border-left: 3px solid #2c8d80; }
    .status-chip.complete { border-left-color: #21b891; background: #eafaf7; }
    .status-chip.pending { border-left-color: #b2a96a; background: #fffdf7; }
    .status-dot { width: 8px; height: 8px; border-radius: 50%; background: #2cae94; display: inline-block; }
    .status-dot.pending { background: #b8a457; }
    .subtle { color: #62797e; font-size: 14px; }
    .main-heading { font-size: 30px; font-weight: 800; color: #20373f; margin: 0 0 10px; }
    .section-title { font-size: 26px; font-weight: 700; color: #20373f; margin: 12px 0; }
    .download-button { background: #0f766e; color: white; border-radius: 9px; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title-panel'><h1>Medallion Pipeline</h1><p>Transform retail data into actionable business insights, everything connected.</p></div>", unsafe_allow_html=True)

if "workflow_status" not in st.session_state:
    st.session_state["workflow_status"] = [
        {"layer": "Bronze layer", "state": "not_started"},
        {"layer": "Silver layer", "state": "not_started"},
        {"layer": "Gold layer", "state": "not_started"},
    ]

left_col, center_col = st.columns([1, 4])

with left_col:
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='control-title'>Run ID</div>", unsafe_allow_html=True)
    run_id = st.text_input("Run ID", value=f"run-{uuid.uuid4().hex[:8]}", label_visibility="collapsed")
    st.markdown("<div class='control-title'>Upload CSV files</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for uploaded in uploaded_files:
            destination = LANDING_DIR / uploaded.name
            destination.write_bytes(uploaded.getvalue())
        st.success(f"Saved {len(uploaded_files)} uploaded file(s) to {LANDING_DIR}")
    st.markdown("</div>", unsafe_allow_html=True)

with center_col:
    st.markdown("<div class='center-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='main-heading'>Ask Your Business Question</div>", unsafe_allow_html=True)
    business_question = st.text_area(
        "What would you like to know about your data?",
        value="What is the yearly sales per product and location?",
        height=100,
    )

    st.markdown("<div class='section-title'>Suggested Questions</div>", unsafe_allow_html=True)
    suggested_questions = [
        "What are yearly sales by product?",
        "Which location has the highest sales?",
        "What are sales trends by region and category?",
        "What products are driving yearly sales growth?",
    ]
    question_cols = st.columns(2)
    for i, suggestion in enumerate(suggested_questions):
        with question_cols[i % 2]:
            if st.button(suggestion, key=f"q_{suggestion}", use_container_width=True):
                business_question = suggestion

    if st.session_state.get("workflow_status"):
        workflow_status_placeholder = st.empty()
        workflow_status_placeholder.markdown(
            render_workflow_status_html(st.session_state["workflow_status"]),
            unsafe_allow_html=True,
        )

    if st.button("Run Full Workflow", use_container_width=True):
        if not uploaded_files:
            st.warning("Select at least one CSV file to upload.")
        else:
            file_paths = [str(LANDING_DIR / p.name) for p in uploaded_files]
            if not file_paths:
                file_paths = [str(p) for p in DEFAULT_FILES]
            status = run_stepwise_pipeline(file_paths, run_id)
            st.session_state["workflow_files"] = file_paths
            st.session_state["workflow_plan"] = status

            st.session_state["workflow_status"] = [
                {"layer": "Bronze layer", "state": "in_progress"},
                {"layer": "Silver layer", "state": "in_progress"},
                {"layer": "Gold layer", "state": "in_progress"},
            ]
            workflow_status_placeholder.markdown(render_workflow_status_html(st.session_state["workflow_status"]), unsafe_allow_html=True)

            notification_placeholder = st.empty()
            notification_placeholder.info("Running profiler and full pipeline workflow...")

            try:
                profile_path = profile_multiple_datasets(file_paths, run_id, "Profile retail CSV inputs")
                bronze_sttm = generate_bronze_sttm(file_paths)
                notification_placeholder.info("Bronze layer in progress")
                bronze_outputs = bronze_ingest(file_paths, output_dir=PROJECT_ROOT / "data" / "bronze_layer")
                st.session_state["workflow_status"][0] = {"layer": "Bronze layer", "state": "completed"}
                workflow_status_placeholder.markdown(render_workflow_status_html(st.session_state["workflow_status"]), unsafe_allow_html=True)
                notification_placeholder.success("Bronze layer completed")

                silver_sttm = generate_silver_sttm(file_paths)
                notification_placeholder.info("Silver layer in progress")
                silver_outputs = silver_clean(file_paths, output_dir=PROJECT_ROOT / "data" / "silver_layer")
                st.session_state["workflow_status"][1] = {"layer": "Silver layer", "state": "completed"}
                workflow_status_placeholder.markdown(render_workflow_status_html(st.session_state["workflow_status"]), unsafe_allow_html=True)
                notification_placeholder.success("Silver layer completed")

                gold_sttm = generate_gold_sttm(file_paths)
                notification_placeholder.info("Gold layer in progress")
                gold_outputs = gold_aggregate(file_paths, output_dir=PROJECT_ROOT / "data" / "gold_layer")
                st.session_state["workflow_status"][2] = {"layer": "Gold layer", "state": "completed"}
                workflow_status_placeholder.markdown(render_workflow_status_html(st.session_state["workflow_status"]), unsafe_allow_html=True)
                notification_placeholder.success("Gold layer completed")

                report = create_report_insight(
                    gold_outputs[0],
                    report_path=PROJECT_ROOT / "reports",
                    business_question=business_question,
                    source_files=file_paths,
                )
                state = run_pipeline_phase("completed", file_paths, run_id)

                st.session_state["profile_path"] = profile_path
                st.session_state["bronze_sttm"] = bronze_sttm
                st.session_state["silver_sttm"] = silver_sttm
                st.session_state["gold_sttm"] = gold_sttm
                st.session_state["bronze_outputs"] = bronze_outputs
                st.session_state["silver_outputs"] = silver_outputs
                st.session_state["gold_outputs"] = gold_outputs
                st.session_state["report"] = report
                st.session_state["state"] = state
                notification_placeholder.success("Pipeline run complete")
            except Exception as exc:
                notification_placeholder.error(f"Pipeline execution failed: {exc}")
                st.stop()

    st.markdown("</div>", unsafe_allow_html=True)

# Only report display remains visible after the run.
if "report" in st.session_state:
    report = st.session_state["report"]
    analysis = report.get("analysis", {})
    st.subheader("Business Question")
    st.write(report.get("business_question"))
    st.subheader("Report Insights")
    for item in report.get("insights", []):
        st.write("•", item)

    if analysis.get("yearly_sales_by_product"):
        st.subheader("Yearly Sales by Product")
        product_df = pd.DataFrame(analysis["yearly_sales_by_product"])
        if {"year", "product_name", "total_amount"}.issubset(set(product_df.columns)):
            product_df = product_df.rename(columns={"total_amount": "sales"})
            product_table_df = product_df[["year", "product_name", "sales"]]
            st.dataframe(product_table_df, use_container_width=True)

    if analysis.get("yearly_sales_by_location"):
        st.subheader("Yearly Sales by Location")
        location_df = pd.DataFrame(analysis["yearly_sales_by_location"])
        if {"year", "region", "city", "state", "total_amount"}.issubset(set(location_df.columns)):
            location_df = location_df.rename(columns={"total_amount": "sales"})
            location_table_df = location_df[["year", "region", "city", "state", "sales"]]
            st.dataframe(location_table_df, use_container_width=True)

    pdf_path = PROJECT_ROOT / "reports" / "report_insights.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download PDF Report",
                data=f.read(),
                file_name="report_insights.pdf",
                mime="application/pdf",
                key="download_pdf_report",
            )
