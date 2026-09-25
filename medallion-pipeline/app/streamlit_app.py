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
from agents.profiler import profile_multiple_datasets
from agents.reporter import create_report_insight
from agents.silver import silver_clean
from agents.sttm import generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm
from core.state import PipelineState

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze_layer"
SILVER_DIR = PROJECT_ROOT / "data" / "silver_layer"
GOLD_DIR = PROJECT_ROOT / "data" / "gold_layer"
REPORTS_DIR = PROJECT_ROOT / "reports"
STTM_DIR = PROJECT_ROOT / "data" / "sttm"

DEFAULT_FILES = [
    LANDING_DIR / "sales_data.csv",
    LANDING_DIR / "products.csv",
    LANDING_DIR / "stores.csv",
]


def _save_sttm(sttm_dict: dict, run_id: str, phase: str) -> str:
    STTM_DIR.mkdir(parents=True, exist_ok=True)
    path = STTM_DIR / f"{phase}_sttm_{run_id}.json"
    path.write_text(json.dumps(sttm_dict, indent=2), encoding="utf-8")
    return str(path)


def render_workflow_status_html(workflow_status: list[dict[str, str]]) -> str:
    rows = []
    for item in workflow_status:
        layer = item["layer"]
        state = item["state"]
        if state == "completed":
            label, cls, dot = f"{layer} completed ✓", "complete", "status-dot"
        elif state == "approved":
            label, cls, dot = f"{layer} approved ✓", "complete", "status-dot"
        elif state == "in_progress":
            label, cls, dot = f"{layer} in progress…", "pending", "status-dot pending"
        else:
            label, cls, dot = layer, "pending", "status-dot pending"
        rows.append(f"<div class='status-chip {cls}'><span class='{dot}'></span>{label}</div>")
    return (
        "<div class='workflow-status'>"
        "<span class='subtle'>STTM Approval Pipeline</span>"
        + "".join(rows)
        + "</div>"
    )


def summarize_files(file_paths: list[str]) -> pd.DataFrame:
    rows = []
    for path in file_paths:
        p = Path(path)
        try:
            df = pd.read_csv(p)
            rows.append({"file": p.name, "rows": len(df), "columns": len(df.columns), "status": "ready"})
        except Exception:
            rows.append({"file": p.name, "rows": 0, "columns": 0, "status": "error"})
    return pd.DataFrame(rows)


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Medallion Pipeline", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .block-container{padding-top:1rem;background:linear-gradient(135deg,#eef8f6 0%,#f8fafb 100%);max-width:1280px;margin:0 auto}
    .title-panel{background:linear-gradient(135deg,#102c2f 0%,#204c54 100%);color:#eef8f8;padding:26px 30px;border-radius:14px;box-shadow:0 8px 22px rgba(16,40,47,.2);margin-bottom:14px}
    .title-panel h1{color:#eef8f8;font-size:clamp(2.2rem,3vw,3rem);margin:0 0 12px}
    .title-panel p{color:#d7f5ed;font-size:1.1rem;margin:0}
    .left-panel{background:#eef7f7;border:1px solid #bdd7d6;border-radius:12px;padding:18px 16px;box-shadow:0 8px 20px rgba(0,0,0,.04)}
    .center-panel{background:#ffffff;border:1px solid #d8e7e9;border-radius:14px;padding:24px;box-shadow:0 12px 30px rgba(0,0,0,.08)}
    .control-title{font-size:17px;font-weight:700;color:#20333a;margin-bottom:10px}
    .stButton button{background:#eef5ee;color:#203d3c;border-radius:12px;border:1px solid #b9c9cb;padding:10px 14px;font-weight:700}
    .workflow-status{margin-top:14px;padding:12px;border-radius:12px;border:1px solid #bfd8d8;background:#eefbfa}
    .status-chip{display:flex;align-items:center;gap:9px;padding:7px 11px;margin-top:8px;border-radius:10px;background:#eaf7f5;color:#183d3d;font-size:12px;border-left:3px solid #2c8d80}
    .status-chip.complete{border-left-color:#21b891;background:#eafaf7}
    .status-chip.pending{border-left-color:#b2a96a;background:#fffdf7}
    .status-dot{width:8px;height:8px;border-radius:50%;background:#2cae94;display:inline-block}
    .status-dot.pending{background:#b8a457}
    .subtle{color:#62797e;font-size:14px}
    .main-heading{font-size:30px;font-weight:800;color:#20373f;margin:0 0 10px}
    .section-title{font-size:22px;font-weight:700;color:#20373f;margin:12px 0}
    .sttm-gate{border:2px solid #0f766e;border-radius:12px;padding:20px;background:#f0fdf8;margin:16px 0}
    .sttm-gate h3{color:#0f766e;margin:0 0 8px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div class='title-panel'><h1>Medallion Pipeline</h1>"
    "<p>Human-in-the-loop STTM approval gates at every layer. "
    "Each transformation requires your review before it executes.</p></div>",
    unsafe_allow_html=True,
)

# ── Session state init ─────────────────────────────────────────────────────
_STAGE_INIT = "init"
_STAGE_BRONZE_REVIEW = "bronze_sttm_review"
_STAGE_SILVER_REVIEW = "silver_sttm_review"
_STAGE_GOLD_REVIEW = "gold_sttm_review"
_STAGE_COMPLETE = "complete"

if "pipeline_stage" not in st.session_state:
    st.session_state["pipeline_stage"] = _STAGE_INIT
if "workflow_status" not in st.session_state:
    st.session_state["workflow_status"] = [
        {"layer": "Bronze layer", "state": "not_started"},
        {"layer": "Silver layer", "state": "not_started"},
        {"layer": "Gold layer", "state": "not_started"},
    ]
if "run_id" not in st.session_state:
    st.session_state["run_id"] = f"run-{uuid.uuid4().hex[:8]}"

# ── Layout ─────────────────────────────────────────────────────────────────
left_col, center_col = st.columns([1, 4])

with left_col:
    st.markdown("<div class='left-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='control-title'>Run ID</div>", unsafe_allow_html=True)
    run_id = st.text_input(
        "Run ID",
        value=st.session_state["run_id"],
        label_visibility="collapsed",
        disabled=st.session_state["pipeline_stage"] != _STAGE_INIT,
    )

    st.markdown("<div class='control-title'>Upload CSV files</div>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload CSV files", type=["csv"], accept_multiple_files=True,
        label_visibility="collapsed",
        disabled=st.session_state["pipeline_stage"] != _STAGE_INIT,
    )
    if uploaded_files and st.session_state["pipeline_stage"] == _STAGE_INIT:
        for uploaded in uploaded_files:
            (LANDING_DIR / uploaded.name).write_bytes(uploaded.getvalue())
        st.success(f"Saved {len(uploaded_files)} file(s)")

    st.markdown("---")
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        for key in [
            "pipeline_stage", "bronze_sttm", "bronze_outputs",
            "silver_sttm", "silver_outputs", "gold_sttm",
            "gold_outputs", "report", "profile_path", "file_paths",
            "business_question",
        ]:
            st.session_state.pop(key, None)
        st.session_state["workflow_status"] = [
            {"layer": "Bronze layer", "state": "not_started"},
            {"layer": "Silver layer", "state": "not_started"},
            {"layer": "Gold layer", "state": "not_started"},
        ]
        st.session_state["pipeline_stage"] = _STAGE_INIT
        st.session_state["run_id"] = f"run-{uuid.uuid4().hex[:8]}"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with center_col:
    st.markdown("<div class='center-panel'>", unsafe_allow_html=True)

    stage = st.session_state["pipeline_stage"]

    # ── Business question (editable only in init stage) ──────────────────
    if stage == _STAGE_INIT:
        st.markdown("<div class='main-heading'>Ask Your Business Question</div>", unsafe_allow_html=True)
        business_question = st.text_area(
            "Business question",
            value=st.session_state.get("business_question", "What is the yearly sales per product and location?"),
            height=100,
            label_visibility="collapsed",
        )
        st.markdown("<div class='section-title'>Suggested Questions</div>", unsafe_allow_html=True)
        q_cols = st.columns(2)
        suggestions = [
            "What are yearly sales by product?",
            "Which location has the highest sales?",
            "What are sales trends by region and category?",
            "What products are driving yearly sales growth?",
        ]
        for i, suggestion in enumerate(suggestions):
            with q_cols[i % 2]:
                if st.button(suggestion, key=f"q_{i}", use_container_width=True):
                    st.session_state["business_question"] = suggestion
                    st.rerun()
    else:
        business_question = st.session_state.get("business_question", "")
        st.info(f"**Business Question:** {business_question}")

    # ── Workflow status bar ───────────────────────────────────────────────
    st.markdown(
        render_workflow_status_html(st.session_state["workflow_status"]),
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════════════
    # STAGE: init — profile data and generate Bronze STTM
    # ══════════════════════════════════════════════════════════════════════
    if stage == _STAGE_INIT:
        if st.button("🚀 Run Full Workflow", use_container_width=True):
            file_paths = (
                [str(LANDING_DIR / u.name) for u in uploaded_files]
                if uploaded_files
                else [str(p) for p in DEFAULT_FILES if p.exists()]
            )
            if not file_paths:
                st.warning("No CSV files found. Upload files or ensure default landing data exists.")
                st.stop()

            with st.spinner("Profiling raw data…"):
                try:
                    profile_path = profile_multiple_datasets(file_paths, run_id, "Profile retail CSV inputs")
                except Exception:
                    profile_path = ""

            with st.spinner("LLM is generating Bronze STTM…"):
                bronze_sttm = generate_bronze_sttm(file_paths)

            st.session_state.update({
                "run_id": run_id,
                "file_paths": file_paths,
                "business_question": business_question,
                "profile_path": profile_path,
                "bronze_sttm": bronze_sttm,
                "pipeline_stage": _STAGE_BRONZE_REVIEW,
            })
            _save_sttm(bronze_sttm, run_id, "bronze")
            st.session_state["workflow_status"][0] = {"layer": "Bronze layer", "state": "in_progress"}
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE: bronze_sttm_review — show STTM, require approval to ingest
    # ══════════════════════════════════════════════════════════════════════
    elif stage == _STAGE_BRONZE_REVIEW:
        st.markdown("<div class='sttm-gate'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Step 1 of 3 — Bronze Layer STTM Approval")
        st.write(
            "The LLM has analyzed your source CSV schemas and produced these ingestion rules. "
            "Review them below. **Approve** to ingest your data into the Bronze layer, "
            "or **Revise** to restart."
        )
        col_json, col_action = st.columns([3, 1])
        with col_json:
            st.json(st.session_state.get("bronze_sttm", {}))
        with col_action:
            st.write("**Your Decision**")
            approve_bronze = st.button("✅ Approve & Ingest", key="approve_bronze", use_container_width=True)
            revise_bronze = st.button("❌ Revise & Restart", key="revise_bronze", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if revise_bronze:
            st.session_state["pipeline_stage"] = _STAGE_INIT
            st.session_state["workflow_status"] = [
                {"layer": "Bronze layer", "state": "not_started"},
                {"layer": "Silver layer", "state": "not_started"},
                {"layer": "Gold layer", "state": "not_started"},
            ]
            st.rerun()

        if approve_bronze:
            # Set hitl_approved in state
            pipeline_state = PipelineState(run_id=st.session_state["run_id"])
            pipeline_state.hitl_approved = True

            with st.spinner("Running Bronze ingestion (CSV → Parquet)…"):
                bronze_outputs = bronze_ingest(
                    st.session_state["file_paths"],
                    output_dir=BRONZE_DIR,
                )
            with st.spinner("LLM is generating Silver STTM from Bronze schema…"):
                silver_sttm = generate_silver_sttm(
                    bronze_outputs,
                    st.session_state.get("business_question", ""),
                )

            st.session_state["bronze_outputs"] = bronze_outputs
            st.session_state["silver_sttm"] = silver_sttm
            _save_sttm(silver_sttm, run_id, "silver")
            st.session_state["workflow_status"][0] = {"layer": "Bronze layer", "state": "completed"}
            st.session_state["workflow_status"][1] = {"layer": "Silver layer", "state": "in_progress"}
            st.session_state["pipeline_stage"] = _STAGE_SILVER_REVIEW
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE: silver_sttm_review — show STTM, require approval to clean
    # ══════════════════════════════════════════════════════════════════════
    elif stage == _STAGE_SILVER_REVIEW:
        bronze_outputs = st.session_state.get("bronze_outputs", [])
        st.success(f"✅ Bronze complete — {len(bronze_outputs)} Parquet file(s) written to `data/bronze_layer/`")

        st.markdown("<div class='sttm-gate'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Step 2 of 3 — Silver Layer STTM Approval")
        st.write(
            "The LLM has analyzed the Bronze Parquet schema and your business question to produce "
            "these cleaning and standardization rules. **Approve** to clean the Bronze data into "
            "the Silver layer, or **Revise** to restart."
        )
        col_json, col_action = st.columns([3, 1])
        with col_json:
            st.json(st.session_state.get("silver_sttm", {}))
        with col_action:
            st.write("**Your Decision**")
            approve_silver = st.button("✅ Approve & Clean", key="approve_silver", use_container_width=True)
            revise_silver = st.button("❌ Revise & Restart", key="revise_silver", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if revise_silver:
            st.session_state["pipeline_stage"] = _STAGE_INIT
            st.session_state["workflow_status"] = [
                {"layer": "Bronze layer", "state": "not_started"},
                {"layer": "Silver layer", "state": "not_started"},
                {"layer": "Gold layer", "state": "not_started"},
            ]
            st.rerun()

        if approve_silver:
            pipeline_state = PipelineState(run_id=st.session_state["run_id"])
            pipeline_state.hitl_approved = True

            with st.spinner("Running Silver cleaning (Bronze Parquet → cleaned Parquet)…"):
                silver_outputs = silver_clean(
                    bronze_outputs,
                    output_dir=SILVER_DIR,
                    business_intent=st.session_state.get("business_question", ""),
                )
            with st.spinner("LLM is generating Gold STTM from Silver schema…"):
                gold_sttm = generate_gold_sttm(
                    silver_outputs,
                    st.session_state.get("business_question", ""),
                )

            st.session_state["silver_outputs"] = silver_outputs
            st.session_state["gold_sttm"] = gold_sttm
            _save_sttm(gold_sttm, run_id, "gold")
            st.session_state["workflow_status"][1] = {"layer": "Silver layer", "state": "completed"}
            st.session_state["workflow_status"][2] = {"layer": "Gold layer", "state": "in_progress"}
            st.session_state["pipeline_stage"] = _STAGE_GOLD_REVIEW
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE: gold_sttm_review — show STTM, require approval to aggregate
    # ══════════════════════════════════════════════════════════════════════
    elif stage == _STAGE_GOLD_REVIEW:
        silver_outputs = st.session_state.get("silver_outputs", [])
        st.success(f"✅ Silver complete — {len(silver_outputs)} Parquet file(s) written to `data/silver_layer/`")

        st.markdown("<div class='sttm-gate'>", unsafe_allow_html=True)
        st.markdown("### 🔍 Step 3 of 3 — Gold Layer STTM Approval")
        st.write(
            "The LLM has analyzed the Silver Parquet schema and produced these KPI aggregation rules "
            "aligned to your business question. **Approve** to compute Gold-layer KPIs and generate "
            "the executive report, or **Revise** to restart."
        )
        col_json, col_action = st.columns([3, 1])
        with col_json:
            st.json(st.session_state.get("gold_sttm", {}))
        with col_action:
            st.write("**Your Decision**")
            approve_gold = st.button("✅ Approve & Generate Report", key="approve_gold", use_container_width=True)
            revise_gold = st.button("❌ Revise & Restart", key="revise_gold", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if revise_gold:
            st.session_state["pipeline_stage"] = _STAGE_INIT
            st.session_state["workflow_status"] = [
                {"layer": "Bronze layer", "state": "not_started"},
                {"layer": "Silver layer", "state": "not_started"},
                {"layer": "Gold layer", "state": "not_started"},
            ]
            st.rerun()

        if approve_gold:
            pipeline_state = PipelineState(run_id=st.session_state["run_id"])
            pipeline_state.hitl_approved = True

            with st.spinner("Running Gold aggregation (Silver Parquet → KPI Parquet)…"):
                gold_outputs = gold_aggregate(
                    silver_outputs,
                    output_dir=GOLD_DIR,
                    business_intent=st.session_state.get("business_question", ""),
                )
            with st.spinner("Generating executive report…"):
                report = create_report_insight(
                    gold_outputs,
                    report_path=REPORTS_DIR,
                    business_question=st.session_state.get("business_question", ""),
                    source_files=st.session_state.get("file_paths", []),
                )

            st.session_state["gold_outputs"] = gold_outputs
            st.session_state["report"] = report
            st.session_state["workflow_status"][2] = {"layer": "Gold layer", "state": "completed"}
            st.session_state["pipeline_stage"] = _STAGE_COMPLETE
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════
    # STAGE: complete
    # ══════════════════════════════════════════════════════════════════════
    elif stage == _STAGE_COMPLETE:
        gold_outputs = st.session_state.get("gold_outputs", [])
        st.success(
            f"✅ Pipeline complete — all three STTM gates approved. "
            f"{len(gold_outputs)} Gold Parquet file(s) written."
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ── Report display (always visible once complete) ─────────────────────────
if st.session_state.get("pipeline_stage") == _STAGE_COMPLETE and "report" in st.session_state:
    st.divider()
    report = st.session_state["report"]
    analysis = report.get("analysis", {})

    st.subheader("📊 Business Question")
    st.write(report.get("business_question"))

    st.subheader("💡 Key Insights")
    for item in report.get("insights", []):
        st.write("•", item)

    if analysis.get("yearly_sales_by_product"):
        st.subheader("📈 Yearly Sales by Product")
        df = pd.DataFrame(analysis["yearly_sales_by_product"])
        if {"year", "product_name", "total_amount"}.issubset(df.columns):
            st.dataframe(
                df.rename(columns={"total_amount": "sales"})[["year", "product_name", "sales"]],
                use_container_width=True,
            )

    if analysis.get("yearly_sales_by_location"):
        st.subheader("📍 Yearly Sales by Location")
        df = pd.DataFrame(analysis["yearly_sales_by_location"])
        if {"year", "region", "city", "state", "total_amount"}.issubset(df.columns):
            st.dataframe(
                df.rename(columns={"total_amount": "sales"})[["year", "region", "city", "state", "sales"]],
                use_container_width=True,
            )

    if analysis.get("yearly_revenue"):
        st.subheader("📊 Yearly Revenue")
        rev_df = pd.DataFrame(analysis["yearly_revenue"])
        st.bar_chart(rev_df.set_index("year")["revenue"])

    pdf_path = REPORTS_DIR / "report_insights.pdf"
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f.read(),
                file_name="report_insights.pdf",
                mime="application/pdf",
                key="download_pdf",
            )
