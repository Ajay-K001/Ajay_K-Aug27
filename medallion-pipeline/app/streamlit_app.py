from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.bronze import bronze_ingest
from agents.gold import gold_aggregate
from agents.profiler import profile_multiple_datasets
from agents.reporter import create_report_insight
from agents.silver import silver_clean
from agents.sttm import (
    generate_bronze_sttm, generate_silver_sttm, generate_gold_sttm, save_sttm_csv,
)
from core.state import PipelineState

LANDING_DIR = PROJECT_ROOT / "data" / "landing"
BRONZE_DIR  = PROJECT_ROOT / "data" / "bronze_layer"
SILVER_DIR  = PROJECT_ROOT / "data" / "silver_layer"
GOLD_DIR    = PROJECT_ROOT / "data" / "gold_layer"
REPORTS_DIR = PROJECT_ROOT / "reports"
STTM_DIR    = PROJECT_ROOT / "data" / "sttm"

DEFAULT_FILES = [
    LANDING_DIR / "sales_data.csv",
    LANDING_DIR / "products.csv",
    LANDING_DIR / "stores.csv",
]

TOTAL_STEPS = 11

_STAGE_INIT         = "init"
_STAGE_BRONZE_GATE  = "bronze_gate"
_STAGE_SILVER_GATE  = "silver_gate"
_STAGE_GOLD_GATE    = "gold_gate"
_STAGE_COMPLETE     = "complete"

STAGE_STEP = {
    _STAGE_INIT:        0,
    _STAGE_BRONZE_GATE: 3,
    _STAGE_SILVER_GATE: 6,
    _STAGE_GOLD_GATE:   9,
    _STAGE_COMPLETE:    11,
}

GATE_COLORS = {
    "Bronze": "#cd7f32",
    "Silver": "#a8b2c1",
    "Gold":   "#f5c842",
}

st.set_page_config(
    page_title="Intent-Driven Agentic Medallion Pipeline",
    page_icon="🏅",
    layout="wide",
)

st.markdown("""
<style>
/* ── Base ── */
body, .stApp { background: #0d1117; color: #e6edf3; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22;
    border-right: 1px solid #21262d;
}
.run-id-box {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    color: #58a6ff;
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 6px 10px;
    word-break: break-all;
    margin-top: 4px;
}

/* ── Pipeline step tracker ── */
.step-track { display: flex; flex-direction: column; gap: 6px; margin: 12px 0; }
.step-row { display: flex; align-items: center; gap: 10px; padding: 6px 8px; border-radius: 8px; }
.step-row.done  { background: #0d2818; border: 1px solid #238636; }
.step-row.active{ background: #1a1f2e; border: 1px solid #388bfd; }
.step-row.wait  { background: transparent; border: 1px solid #21262d; opacity: 0.5; }
.step-icon { font-size: 16px; width: 24px; text-align: center; }
.step-name { font-size: 12px; color: #e6edf3; font-weight: 600; }
.step-state{ font-size: 10px; color: #8b949e; margin-left: auto; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, #238636, #2ea043) !important; border-radius: 4px !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d2137 0%, #0f3460 50%, #162447 100%);
    border: 1px solid #1f4e79;
    border-radius: 14px;
    padding: 32px 36px;
    margin-bottom: 24px;
}
.hero h1 { color: #e6edf3; font-size: 28px; margin: 0 0 8px 0; font-weight: 800; }
.hero p  { color: #8b949e; font-size: 14px; margin: 0; }
.hero .badge {
    display: inline-block;
    background: #1f4e79;
    color: #58a6ff;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* ── Input cards ── */
.input-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 22px;
    height: 100%;
}
.input-card h3 { color: #e6edf3; font-size: 15px; margin: 0 0 12px 0; font-weight: 700; }
.input-card p  { color: #8b949e; font-size: 12px; margin: 0 0 10px 0; }

/* ── HITL gate card ── */
.gate-card {
    border-radius: 12px;
    padding: 28px;
    margin-bottom: 20px;
}
.gate-header-bronze { border-left: 4px solid #cd7f32; background: #1a1209; border: 1px solid #3d2b00; border-left: 4px solid #cd7f32; }
.gate-header-silver { border-left: 4px solid #a8b2c1; background: #131820; border: 1px solid #2d3748; border-left: 4px solid #a8b2c1; }
.gate-header-gold   { border-left: 4px solid #f5c842; background: #1a1800; border: 1px solid #3d3600; border-left: 4px solid #f5c842; }
.gate-num   { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.gate-title { font-size: 22px; font-weight: 800; color: #e6edf3; margin-bottom: 4px; }
.gate-sub   { font-size: 13px; color: #8b949e; margin-bottom: 14px; }
.gate-meta  { font-family: monospace; font-size: 11px; color: #6e7681; background: #0d1117; padding: 6px 10px; border-radius: 6px; margin-bottom: 14px; }
.gate-pill  {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
    margin-right: 8px;
    margin-bottom: 16px;
}
.pill-obj  { background: #0d2818; color: #3fb950; border: 1px solid #238636; }
.pill-rule { background: #0d1b2e; color: #58a6ff; border: 1px solid #1f6feb; }
.pill-note { background: #1a1209; color: #d29922; border: 1px solid #9e6a03; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0d1117; border-radius: 8px; gap: 4px; }
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8b949e; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #e6edf3 !important; }

/* ── Obj/Note items ── */
.obj-item  { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 14px; color: #c9d1d9; line-height: 1.5; }
.note-item { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 14px; color: #c9d1d9; line-height: 1.5; }

/* ── All buttons: universal dark text fix ── */
button, .stButton > button, [data-testid*="Button"] > button,
[data-testid*="button"] > button {
    color: #e6edf3 !important;
    font-weight: 600 !important;
}

/* ── Primary (Run Pipeline) ── */
[data-testid="stBaseButton-primary"],
button[kind="primary"],
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e05c25, #c94a15) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* ── Secondary (all other buttons) ── */
[data-testid="stBaseButton-secondary"],
button[kind="secondary"],
.stButton > button[kind="secondary"],
.stButton > button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #444c56 !important;
    border-radius: 8px !important;
}

/* ── Approve button override ── */
.approve-btn [data-testid="stBaseButton-secondary"],
.approve-btn button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}
.approve-btn button p,
.approve-btn button span,
.approve-btn [data-testid="stBaseButton-secondary"] p,
.approve-btn [data-testid="stBaseButton-secondary"] span {
    color: #ffffff !important;
}

/* ── Reject button override ── */
.reject-btn [data-testid="stBaseButton-secondary"],
.reject-btn button {
    background: #200e0e !important;
    color: #f85149 !important;
    border: 1px solid #6e1f1f !important;
    font-weight: 600 !important;
}
.reject-btn button p,
.reject-btn button span,
.reject-btn [data-testid="stBaseButton-secondary"] p,
.reject-btn [data-testid="stBaseButton-secondary"] span {
    color: #f85149 !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] button,
.stDownloadButton > button {
    background: #21262d !important;
    color: #c9d1d9 !important;
    border: 1px solid #444c56 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] button:hover {
    background: #2d333b !important;
    color: #79c0ff !important;
    border-color: #388bfd !important;
}

/* ── Reset Pipeline button ── */
section[data-testid="stSidebar"] button {
    background: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #444c56 !important;
    border-radius: 8px !important;
}

/* ── Metric tiles ── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
}
.metric-label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.metric-value { font-size: 26px; font-weight: 800; color: #e6edf3; }
.metric-sub   { font-size: 11px; color: #3fb950; margin-top: 4px; }

/* ── Insights card ── */
.insights-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #238636;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 20px;
}
.insights-card h3 { color: #3fb950; font-size: 14px; text-transform: uppercase; letter-spacing: 0.8px; margin: 0 0 14px 0; }
.insight-row { padding: 7px 0; border-bottom: 1px solid #21262d; color: #c9d1d9; font-size: 14px; }
.insight-row:last-child { border-bottom: none; }

/* ── Section headings ── */
.section-head {
    font-size: 16px;
    font-weight: 700;
    color: #e6edf3;
    margin: 24px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
}

/* ── Status banner ── */
.status-success {
    background: #0d2818;
    border: 1px solid #238636;
    border-radius: 10px;
    padding: 12px 18px;
    color: #3fb950;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 16px;
}
.status-info {
    background: #0d1b2e;
    border: 1px solid #1f6feb;
    border-radius: 10px;
    padding: 12px 18px;
    color: #58a6ff;
    font-size: 14px;
    font-weight: 600;
    margin-bottom: 16px;
}

/* ── Download section ── */
.dl-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 22px;
    margin-top: 24px;
}
.dl-card h3 { color: #e6edf3; font-size: 15px; margin: 0 0 14px 0; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Fix textarea / text input visibility ── */
textarea, .stTextArea textarea {
    background-color: #1c2128 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    caret-color: #e6edf3 !important;
}
textarea::placeholder, .stTextArea textarea::placeholder {
    color: #6e7681 !important;
}
textarea:focus, .stTextArea textarea:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.2) !important;
}
input[type="text"], .stTextInput input {
    background-color: #1c2128 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "pipeline_stage": _STAGE_INIT,
        "run_id": f"run-{uuid.uuid4().hex[:8]}",
        "file_paths": [],
        "business_question": "",
        "bronze_sttm": {},
        "bronze_outputs": [],
        "silver_sttm": {},
        "silver_outputs": [],
        "gold_sttm": {},
        "gold_outputs": [],
        "report": {},
        "current_step": 0,
        "status_text": "No active run yet.",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


def _step(n: int, text: str):
    st.session_state["current_step"] = n
    st.session_state["status_text"] = text


def _save_sttm_files(sttm: dict, phase: str, run_id: str) -> str:
    STTM_DIR.mkdir(parents=True, exist_ok=True)
    csv_path  = STTM_DIR / f"sttm_{phase}_{run_id}.csv"
    json_path = STTM_DIR / f"sttm_{phase}_{run_id}.json"
    save_sttm_csv(sttm, csv_path)
    json_path.write_text(json.dumps(sttm, indent=2), encoding="utf-8")
    return str(csv_path)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    stage  = st.session_state["pipeline_stage"]
    run_id = st.session_state["run_id"]
    step   = st.session_state["current_step"]
    status = st.session_state["status_text"]

    st.markdown("### 🏅 Pipeline Status")
    st.markdown(f"<div class='run-id-box'>{run_id}</div>", unsafe_allow_html=True)
    st.markdown("")

    # Step tracker
    layers = [
        ("📁", "Data Profiling",  step >= 2,  stage == _STAGE_INIT and step > 0),
        ("🥉", "Bronze Layer",    step >= 5,  stage == _STAGE_BRONZE_GATE),
        ("🥈", "Silver Layer",    step >= 8,  stage == _STAGE_SILVER_GATE),
        ("🥇", "Gold Layer",      step >= 10, stage == _STAGE_GOLD_GATE),
        ("📊", "Report",          step == 11, False),
    ]
    html_steps = "<div class='step-track'>"
    for icon, name, done, active in layers:
        cls  = "done" if done else ("active" if active else "wait")
        flag = "✓" if done else ("⟳" if active else "·")
        html_steps += f"<div class='step-row {cls}'><span class='step-icon'>{icon}</span><span class='step-name'>{name}</span><span class='step-state'>{flag}</span></div>"
    html_steps += "</div>"
    st.markdown(html_steps, unsafe_allow_html=True)

    st.markdown(f"**Progress:** {step}/{TOTAL_STEPS} steps")
    st.progress(step / TOTAL_STEPS)
    st.caption(f"Status: {status}")

    st.divider()
    if st.button("🔄 Reset Pipeline", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='badge'>🤖 AGENTIC · MEDALLION ARCHITECTURE · HITL APPROVED</div>
  <h1>Intent-Driven Agentic Medallion Pipeline</h1>
  <p>Upload your CSV data, define a business question, and let the AI-driven pipeline ingest, clean, aggregate, and report — with human approval at every layer.</p>
</div>
""", unsafe_allow_html=True)

stage = st.session_state["pipeline_stage"]


# ── Gate renderer ─────────────────────────────────────────────────────────────
def _render_gate(gate_num: int, layer: str, sttm: dict, approve_label: str, approve_key: str, reject_key: str, csv_path: str = ""):
    rules      = sttm.get("rules", [])
    objectives = sttm.get("objectives", [])
    notes      = sttm.get("notes", [])
    color      = GATE_COLORS.get(layer, "#58a6ff")
    css_class  = f"gate-header-{layer.lower()}"

    st.markdown(f"""
    <div class='gate-card {css_class}'>
      <div class='gate-num' style='color:{color}'>HITL APPROVAL · GATE {gate_num} OF 3</div>
      <div class='gate-title'>Approve {layer} STTM</div>
      <div class='gate-sub'>Review the Semantic Table Type Mapping before the pipeline proceeds to the {layer} layer.</div>
      <div class='gate-meta'>Run ID: {st.session_state['run_id']}{(" &nbsp;|&nbsp; STTM: " + csv_path) if csv_path else ""}</div>
      <span class='gate-pill pill-obj'>✦ {len(objectives)} Objectives</span>
      <span class='gate-pill pill-rule'>⊞ {len(rules)} Rules</span>
      <span class='gate-pill pill-note'>✎ {len(notes)} Notes</span>
    </div>
    """, unsafe_allow_html=True)

    log_entries = sttm.get("log", [])
    tab_obj, tab_rules, tab_notes, tab_log = st.tabs(["📋 Objectives", "📐 Rules", "📝 Notes", "🗒 Log"])
    with tab_obj:
        for i, obj in enumerate(objectives, 1):
            st.markdown(f"<div class='obj-item'><b>{i}.</b> {obj}</div>", unsafe_allow_html=True)
    with tab_rules:
        if rules:
            st.dataframe(pd.DataFrame(rules), use_container_width=True, hide_index=True)
        else:
            st.info("No column rules generated.")
    with tab_notes:
        for i, note in enumerate(notes, 1):
            st.markdown(f"<div class='note-item'><b>{i}.</b> {note}</div>", unsafe_allow_html=True)
    with tab_log:
        if log_entries:
            log_text = "\n".join(log_entries)
            st.code(log_text, language="text")
        else:
            st.info("No log entries available.")

    st.markdown("")
    col_approve, col_reject = st.columns([1, 1])
    with col_approve:
        st.markdown("<div class='approve-btn'>", unsafe_allow_html=True)
        approved = st.button(f"✅ {approve_label}", key=approve_key, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_reject:
        st.markdown("<div class='reject-btn'>", unsafe_allow_html=True)
        rejected = st.button("✗ Reject & Start Over", key=reject_key, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    return approved, rejected


# ══════════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════════
if stage == _STAGE_INIT:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        st.markdown("<h3>📂 Source Data Files</h3>", unsafe_allow_html=True)
        st.markdown("<p>Upload one or more CSV files (e.g. sales, products, stores). The pipeline will ingest all uploaded files.</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload CSVs", type=["csv"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            for f in uploaded:
                (LANDING_DIR / f.name).write_bytes(f.getvalue())
            st.markdown(f"<p style='color:#3fb950;font-size:13px;font-weight:600;margin-top:10px;'>✓ {len(uploaded)} file(s) ready</p>", unsafe_allow_html=True)
            for f in uploaded:
                st.markdown(f"<span style='font-size:12px;color:#c9d1d9;'>• {f.name} — {f.size//1024} KB</span><br>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#8b949e;font-size:12px;margin-top:10px;'>⚠ No files uploaded yet. Please upload at least one CSV file.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='input-card'>", unsafe_allow_html=True)
        st.markdown("<h3>💡 Business Question</h3>", unsafe_allow_html=True)
        st.markdown("<p>Describe what you want to discover. The pipeline will tailor the STTM and report to your question.</p>", unsafe_allow_html=True)
        business_q = st.text_area(
            "Business intent",
            value=st.session_state.get("business_question", ""),
            height=140,
            label_visibility="collapsed",
            placeholder="e.g. Which product category generates the most revenue by region and year?",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    if not business_q.strip():
        st.warning("⚠️ Please enter a business question before running.")
        st.stop()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_clicked = st.button("▶ Run Pipeline", type="primary", use_container_width=True)
    with col_info:
        st.markdown("<p style='color:#8b949e;font-size:13px;padding-top:10px;'>Pipeline runs Bronze → Silver → Gold with 3 HITL approval gates. Estimated time: 30–60 seconds per layer.</p>", unsafe_allow_html=True)

    if run_clicked:
        if not uploaded:
            st.error("⚠️ Please upload at least one CSV file before running the pipeline.")
            st.stop()
        file_paths = [str(LANDING_DIR / f.name) for f in uploaded]

        st.session_state["file_paths"] = file_paths
        st.session_state["business_question"] = business_q
        _step(1, "profiling")

        with st.spinner("🔍 Profiling source data and drafting Bronze STTM..."):
            try:
                profile_multiple_datasets(file_paths, run_id, f"Profile: {business_q}")
            except Exception:
                pass
            _step(2, "profiled")
            bronze_sttm = generate_bronze_sttm(file_paths, business_q)
            _step(3, "awaiting_bronze_approval")

        st.session_state["bronze_sttm"] = bronze_sttm
        _save_sttm_files(bronze_sttm, "bronze", run_id)
        st.session_state["pipeline_stage"] = _STAGE_BRONZE_GATE
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# BRONZE GATE
# ══════════════════════════════════════════════════════════════════════════════
elif stage == _STAGE_BRONZE_GATE:
    st.markdown("<div class='status-info'>🔍 Data profiling complete — Bronze STTM generated. Review and approve to begin ingestion.</div>", unsafe_allow_html=True)
    csv_path = str(STTM_DIR / f"sttm_bronze_{run_id}.csv")
    approved, rejected = _render_gate(
        1, "Bronze", st.session_state["bronze_sttm"],
        "Approve Bronze STTM", "approve_bronze", "reject_bronze", csv_path,
    )

    if rejected:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if approved:
        PipelineState(run_id=run_id).hitl_approved = True
        _step(4, "bronze_ingesting")
        with st.spinner("🥉 Ingesting raw data into Bronze layer..."):
            bronze_outputs = bronze_ingest(st.session_state["file_paths"], output_dir=BRONZE_DIR)
            _step(5, "bronze_complete")
            silver_sttm = generate_silver_sttm(bronze_outputs, st.session_state["business_question"])
            _step(6, "awaiting_silver_approval")

        st.session_state["bronze_outputs"] = bronze_outputs
        st.session_state["silver_sttm"] = silver_sttm
        _save_sttm_files(silver_sttm, "silver", run_id)
        st.session_state["pipeline_stage"] = _STAGE_SILVER_GATE
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# SILVER GATE
# ══════════════════════════════════════════════════════════════════════════════
elif stage == _STAGE_SILVER_GATE:
    bronze_outputs = st.session_state.get("bronze_outputs", [])
    st.markdown(f"<div class='status-success'>🥉 Bronze complete — {len(bronze_outputs)} Parquet file(s) written to <code>data/bronze_layer/</code></div>", unsafe_allow_html=True)
    csv_path = str(STTM_DIR / f"sttm_silver_{run_id}.csv")
    approved, rejected = _render_gate(
        2, "Silver", st.session_state["silver_sttm"],
        "Approve Silver STTM", "approve_silver", "reject_silver", csv_path,
    )

    if rejected:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if approved:
        PipelineState(run_id=run_id).hitl_approved = True
        _step(7, "silver_cleaning")
        with st.spinner("🥈 Cleaning and standardising Silver layer..."):
            silver_outputs = silver_clean(
                bronze_outputs,
                output_dir=SILVER_DIR,
                business_intent=st.session_state["business_question"],
            )
            _step(8, "silver_complete")
            gold_sttm = generate_gold_sttm(silver_outputs, st.session_state["business_question"])
            _step(9, "awaiting_gold_approval")

        st.session_state["silver_outputs"] = silver_outputs
        st.session_state["gold_sttm"] = gold_sttm
        _save_sttm_files(gold_sttm, "gold", run_id)
        st.session_state["pipeline_stage"] = _STAGE_GOLD_GATE
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# GOLD GATE
# ══════════════════════════════════════════════════════════════════════════════
elif stage == _STAGE_GOLD_GATE:
    silver_outputs = st.session_state.get("silver_outputs", [])
    st.markdown(f"<div class='status-success'>🥈 Silver complete — {len(silver_outputs)} Parquet file(s) written to <code>data/silver_layer/</code></div>", unsafe_allow_html=True)
    csv_path = str(STTM_DIR / f"sttm_gold_{run_id}.csv")
    approved, rejected = _render_gate(
        3, "Gold", st.session_state["gold_sttm"],
        "Approve Gold STTM", "approve_gold", "reject_gold", csv_path,
    )

    if rejected:
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    if approved:
        PipelineState(run_id=run_id).hitl_approved = True
        _step(10, "gold_aggregating")
        with st.spinner("🥇 Aggregating Gold KPIs and generating executive report..."):
            gold_outputs = gold_aggregate(
                silver_outputs,
                output_dir=GOLD_DIR,
                business_intent=st.session_state["business_question"],
            )
            _step(11, "completed")
            report = create_report_insight(
                gold_outputs,
                report_path=REPORTS_DIR,
                business_question=st.session_state["business_question"],
                source_files=st.session_state["file_paths"],
            )

        st.session_state["gold_outputs"] = gold_outputs
        st.session_state["report"] = report
        st.session_state["pipeline_stage"] = _STAGE_COMPLETE
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# COMPLETE
# ══════════════════════════════════════════════════════════════════════════════
elif stage == _STAGE_COMPLETE:
    gold_outputs = st.session_state.get("gold_outputs", [])
    report   = st.session_state.get("report", {})
    analysis = report.get("analysis", {})
    charts   = report.get("charts", [])

    st.markdown(f"<div class='status-success'>🎉 Pipeline complete — {TOTAL_STEPS}/{TOTAL_STEPS} steps · {len(gold_outputs)} Gold Parquet file(s) · <code>data/gold_layer/</code></div>", unsafe_allow_html=True)

    # ── Business question ──
    bq = report.get("business_question", "")
    if bq:
        st.markdown(f"""
        <div style='background:#0d1b2e;border:1px solid #1f6feb;border-radius:10px;padding:14px 20px;margin-bottom:20px;'>
          <span style='font-size:11px;color:#58a6ff;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;'>Business Question</span><br>
          <span style='font-size:16px;color:#e6edf3;font-weight:600;'>{bq}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Metric tiles ──
    yearly  = analysis.get("yearly_revenue", [])
    top_prods = analysis.get("top_products_by_total_sales", [])
    locations = analysis.get("yearly_sales_by_location", [])

    metrics = {}
    if yearly:
        metrics["Total Revenue"]  = (f"${sum(r.get('revenue',0) for r in yearly):,.0f}", "All years combined")
        metrics["Best Year"]      = (str(max(yearly, key=lambda r: r.get('revenue',0)).get('year','-')), f"${max(r.get('revenue',0) for r in yearly):,.0f}")
    if top_prods:
        metrics["Top Product"]    = (top_prods[0].get('product_name','?')[:20], f"${top_prods[0].get('total_amount',0):,.2f}")
    if locations:
        best_loc = max(locations, key=lambda r: r.get('total_amount',0))
        metrics["Best Location"]  = (best_loc.get('city','?'), best_loc.get('region','?'))

    if metrics:
        cols = st.columns(len(metrics))
        for col, (label, (value, sub)) in zip(cols, metrics.items()):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value'>{value}</div>
                  <div class='metric-sub'>{sub}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("")

    # ── Key insights ──
    insights = report.get("insights", [])
    if insights:
        rows_html = "".join(f"<div class='insight-row'>💡 {i}</div>" for i in insights)
        st.markdown(f"""
        <div class='insights-card'>
          <h3>Key Insights</h3>
          {rows_html}
        </div>
        """, unsafe_allow_html=True)

    # ── Charts ──
    if charts:
        st.markdown("<div class='section-head'>📈 Analysis Charts</div>", unsafe_allow_html=True)
        if len(charts) == 1:
            try:
                fig = pio.from_json(charts[0]["json"])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
        else:
            for i in range(0, len(charts), 2):
                c1, c2 = st.columns(2)
                with c1:
                    try:
                        fig = pio.from_json(charts[i]["json"])
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        pass
                with c2:
                    if i + 1 < len(charts):
                        try:
                            fig = pio.from_json(charts[i+1]["json"])
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            pass

    # ── Data tables ──
    tables_shown = False
    if analysis.get("top_products_by_total_sales"):
        if not tables_shown:
            st.markdown("<div class='section-head'>📋 Data Tables</div>", unsafe_allow_html=True)
            tables_shown = True
        st.markdown("**Top Products by Revenue**")
        df = pd.DataFrame(analysis["top_products_by_total_sales"])
        if "total_amount" in df.columns:
            df = df.rename(columns={"total_amount": "revenue"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    if analysis.get("yearly_sales_by_product"):
        if not tables_shown:
            st.markdown("<div class='section-head'>📋 Data Tables</div>", unsafe_allow_html=True)
            tables_shown = True
        st.markdown("**Yearly Sales by Product**")
        df = pd.DataFrame(analysis["yearly_sales_by_product"])
        if "total_amount" in df.columns:
            df = df.rename(columns={"total_amount": "revenue"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    if analysis.get("yearly_sales_by_location"):
        st.markdown("**Yearly Sales by Location**")
        df = pd.DataFrame(analysis["yearly_sales_by_location"])
        if "total_amount" in df.columns:
            df = df.rename(columns={"total_amount": "revenue"})
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Downloads ──
    st.markdown("""
    <div class='dl-card'>
      <h3>⬇ Download Artifacts</h3>
    </div>
    """, unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        pdf_path = REPORTS_DIR / "report_insights.pdf"
        if pdf_path.exists():
            with open(pdf_path, "rb") as f:
                st.download_button("📥 PDF Report", f.read(), "report_insights.pdf", "application/pdf", use_container_width=True)
    with col2:
        json_path = REPORTS_DIR / "report_insights.json"
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                st.download_button("📄 JSON Report", f.read(), "report_insights.json", "application/json", use_container_width=True)
    with col3:
        html_path = REPORTS_DIR / "report_insights.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                st.download_button("🌐 HTML Report", f.read(), "report_insights.html", "text/html", use_container_width=True)
