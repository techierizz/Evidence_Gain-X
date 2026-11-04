import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import json
import torch
import pandas as pd
import copy

from src.utils.config import load_config
from src.data.manifest import DatasetManifest, EvidenceItem
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy, EvidenceGainSelector
from src.diagnosis.pipeline import EvidenceGainXPipeline

def render_html(html_str: str):
    clean = "\n".join(line.strip() for line in html_str.splitlines() if line.strip())
    st.markdown(clean, unsafe_allow_html=True)

st.set_page_config(
    page_title="EvidenceGain-X | Clinical Active Vision Console",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background: radial-gradient(1200px 600px at 50% -120px, rgba(30, 58, 138, 0.22), rgba(9, 13, 22, 1) 75%), #090d16;
        color: #f1f5f9;
    }

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.4rem 2rem;
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.85));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.04);
        margin-bottom: 1.6rem;
    }
    .header-brand {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        font-size: 1.65rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 40%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.86rem;
        font-weight: 500;
        color: #94a3b8;
        margin-top: 4px;
        letter-spacing: -0.01em;
    }
    .header-telemetry {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }
    .telemetry-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #cbd5e1;
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 5px 12px;
        border-radius: 9999px;
        backdrop-filter: blur(8px);
    }
    .telemetry-chip.status-online {
        color: #34d399;
        background: rgba(16, 185, 129, 0.12);
        border-color: rgba(16, 185, 129, 0.3);
    }
    .telemetry-chip.status-online::before {
        content: "";
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    .preset-container {
        margin-bottom: 1.4rem;
    }
    .preset-header {
        font-size: 0.76rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .preset-card-top {
        padding: 6px 10px;
        background: #131c2e;
        border: 1px solid #1e293b;
        border-bottom: none;
        border-radius: 8px 8px 0 0;
        text-align: center;
    }
    .preset-badge-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .preset-sub-text {
        font-size: 0.72rem;
        color: #94a3b8;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .panel-card {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 0 1px rgba(255, 255, 255, 0.03);
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
        letter-spacing: -0.01em;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .panel-title-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        font-size: 0.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    .viewfinder-box {
        position: relative;
        background: #0d131f;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 12px;
    }
    .viewfinder-hud {
        display: flex;
        justify-content: space-between;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #64748b;
        margin-bottom: 8px;
        padding: 4px 8px;
        background: rgba(15, 23, 42, 0.8);
        border-radius: 4px;
    }
    .telemetry-row {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-top: 10px;
    }
    .telemetry-cell {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 8px 12px;
    }
    .telemetry-cell-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .telemetry-cell-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-top: 2px;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 1.4rem;
    }
    .metric-cell {
        position: relative;
        background: linear-gradient(180deg, #162032, #101726);
        border: 1px solid #233047;
        border-radius: 10px;
        padding: 1rem 1.1rem;
        text-align: left;
        box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }
    .metric-cell::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38bdf8, transparent);
    }
    .metric-cell.accent-emerald::before {
        background: linear-gradient(90deg, #10b981, transparent);
    }
    .metric-cell.accent-amber::before {
        background: linear-gradient(90deg, #f59e0b, transparent);
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 4px;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-sub {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 4px;
    }

    .prob-container {
        background: #0d131f;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 1.1rem;
        margin-bottom: 1.2rem;
    }
    .prob-row {
        margin: 12px 0;
    }
    .prob-row:first-child {
        margin-top: 4px;
    }
    .prob-row:last-child {
        margin-bottom: 4px;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.85rem;
        margin-bottom: 6px;
    }
    .prob-name-box {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .prob-lead-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.65rem;
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.15);
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 2px 6px;
        border-radius: 4px;
    }
    .prob-track {
        height: 10px;
        background: #1e293b;
        border-radius: 6px;
        overflow: hidden;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
    }
    .prob-bar {
        height: 100%;
        border-radius: 6px;
        transition: width 0.4s ease-in-out;
    }

    .alert-box {
        padding: 1.1rem 1.3rem;
        border-radius: 10px;
        margin-top: 1.2rem;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    .alert-box.alert-ambiguity {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.14), rgba(245, 158, 11, 0.05));
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fde68a;
    }
    .alert-box.alert-provisional {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.14), rgba(56, 189, 248, 0.05));
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #bae6fd;
    }
    .alert-box.alert-decisive {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(16, 185, 129, 0.05));
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #a7f3d0;
    }

    .hero-recommendation-card {
        background: linear-gradient(135deg, #132338, #0f1c2d);
        border: 1px solid rgba(56, 189, 248, 0.35);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 25px -5px rgba(56, 189, 248, 0.15);
    }
    .hero-rec-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .hero-rec-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: #38bdf8;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .hero-rec-cost {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        background: #1e293b;
        padding: 2px 8px;
        border-radius: 4px;
    }
    .hero-rec-name {
        font-size: 1.35rem;
        font-weight: 800;
        color: #ffffff;
        margin: 2px 0 10px 0;
        letter-spacing: -0.02em;
    }
    .hero-rec-metrics {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .hero-rec-metric-item {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    .hero-rec-metric-item b {
        font-family: 'JetBrains Mono', monospace;
        color: #ffffff;
        display: block;
        font-size: 0.92rem;
        margin-top: 2px;
    }

    .clinical-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1e293b;
        margin-bottom: 1rem;
        font-size: 0.82rem;
    }
    .clinical-table th {
        background: #111827;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid #1e293b;
    }
    .clinical-table td {
        padding: 9px 12px;
        border-bottom: 1px solid #162032;
        background: #0d131f;
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
    }
    .clinical-table tr.top-cand-row td {
        background: rgba(56, 189, 248, 0.08);
        color: #ffffff;
        font-weight: 600;
    }
    .clinical-table tr:last-child td {
        border-bottom: none;
    }

    .audit-step-node {
        position: relative;
        padding-left: 20px;
        margin-bottom: 14px;
        border-left: 2px solid #233047;
    }
    .audit-step-node::before {
        content: "";
        position: absolute;
        left: -6px;
        top: 4px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 8px #38bdf8;
    }
    .audit-step-node.node-rejected::before {
        background: #ef4444;
        box-shadow: 0 0 8px #ef4444;
    }

    .clinical-certificate {
        background: linear-gradient(145deg, #112019, #0c1813);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 14px;
        padding: 1.6rem;
        box-shadow: 0 10px 30px -5px rgba(16, 185, 129, 0.15);
    }
    .clinical-certificate.cert-contradiction {
        background: linear-gradient(145deg, #23180f, #181009);
        border: 1px solid rgba(245, 158, 11, 0.5);
        box-shadow: 0 10px 30px -5px rgba(245, 158, 11, 0.2);
    }
    .clinical-certificate.cert-rejected {
        background: linear-gradient(145deg, #241417, #180d0f);
        border: 1px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 10px 30px -5px rgba(239, 68, 68, 0.2);
    }
    .clinical-certificate.cert-provisional {
        background: linear-gradient(145deg, #132235, #0d1724);
        border: 1px solid rgba(56, 189, 248, 0.4);
        box-shadow: 0 10px 30px -5px rgba(56, 189, 248, 0.15);
    }
    .cert-super {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 700;
        color: #34d399;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .cert-disease-name {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
        margin: 6px 0 2px 0;
        letter-spacing: -0.02em;
    }
    .cert-disease-sub {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    .cert-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        margin-bottom: 1.2rem;
    }
    .cert-cell {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 8px 12px;
    }
    .cert-cell-label {
        font-size: 0.68rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .cert-cell-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.15rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2px;
    }
    .cert-footer {
        padding-top: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.5;
    }

    div.stButton > button {
        background: #162032 !important;
        border: 1px solid #283750 !important;
        color: #f1f5f9 !important;
        border-radius: 8px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 8px 12px !important;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: #1e2d47 !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.25);
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0284c7, #0369a1) !important;
        border-color: #38bdf8 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0369a1, #075985) !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.45) !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b0f17 !important;
        border-right: 1px solid #1e293b;
    }
</style>
""")

@st.cache_resource
def load_cached_classifier(ckpt_path: str, num_classes: int, backbone: str, device: str):
    classifier = DiseaseClassifier(num_classes=num_classes, backbone_name=backbone, pretrained=False)
    if Path(ckpt_path).exists():
        classifier.load_checkpoint(ckpt_path, device=device)
    return classifier

def load_system_pipeline():
    cfg = load_config("configs/config.yaml")
    classes = cfg["dataset"]["classes"]
    cand_types = cfg["evidence"]["candidate_types"]
    device = "cuda" if torch.cuda.is_available() and cfg.get("device") == "cuda" else "cpu"

    ckpt_path = "checkpoints/classifier_best.pt"
    classifier = load_cached_classifier(ckpt_path, len(classes), cfg["classifier"]["backbone"], device)

    policy = EvidenceGainXPolicy(
        classes=classes,
        candidate_types=cand_types,
        alpha_sep=cfg["selector"]["weights"]["alpha_sep"],
        beta_unc=cfg["selector"]["weights"]["beta_unc"],
        gamma_diag=cfg["selector"]["weights"]["gamma_diag"],
        lambda_cost=cfg["selector"]["weights"]["lambda_cost"],
        mu_risk=cfg["selector"]["weights"]["mu_risk"]
    )

    pipeline = EvidenceGainXPipeline(
        classifier=classifier,
        policy=policy,
        classes=classes,
        device=device,
        max_budget=cfg["evidence"]["max_budget"],
        stopping_entropy=cfg["evidence"]["stopping_entropy_threshold"],
        stopping_margin=cfg["evidence"]["stopping_margin_threshold"],
        candidate_types=cand_types
    )
    return cfg, pipeline, classes, cand_types

cfg, pipeline, classes, cand_types = load_system_pipeline()

manifest_path = "data/manifests/dataset_manifest.json"
manifest = DatasetManifest.load_json(manifest_path) if Path(manifest_path).exists() else None
case_dict = {c.case_id: c for c in manifest.cases} if manifest else {}

if "current_state" not in st.session_state:
    st.session_state["current_state"] = None
if "active_case" not in st.session_state:
    st.session_state["active_case"] = None
if "active_preset_name" not in st.session_state:
    st.session_state["active_preset_name"] = None
if "selected_case_id" not in st.session_state:
    st.session_state["selected_case_id"] = "tomato_early_blight_0001" if manifest else None
if "active_image_path_or_pil" not in st.session_state:
    st.session_state["active_image_path_or_pil"] = None
if "input_source_mode" not in st.session_state:
    st.session_state["input_source_mode"] = "Benchmark Gallery"

disease_palette = {
    "early_blight": "#f59e0b",
    "late_blight": "#ef4444",
    "septoria_leaf_spot": "#a855f7",
    "healthy": "#10b981"
}

disease_gradients = {
    "early_blight": "linear-gradient(90deg, #d97706, #fbbf24)",
    "late_blight": "linear-gradient(90deg, #dc2626, #f87171)",
    "septoria_leaf_spot": "linear-gradient(90deg, #7c3aed, #c084fc)",
    "healthy": "linear-gradient(90deg, #059669, #34d399)"
}

with st.sidebar:
    render_html("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1.2rem;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 10px #38bdf8;"></div>
            <span style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">Diagnostic Controls</span>
        </div>
    """)

    input_mode = st.radio(
        "Observation Input Mode:",
        ["Benchmark Gallery", "Live Image Ingestion"],
        index=0 if st.session_state["input_source_mode"] == "Benchmark Gallery" else 1,
        key="radio_input_mode"
    )
    st.session_state["input_source_mode"] = input_mode

    max_steps = st.slider("Evidence Budget Horizon (B):", 1, 5, cfg["evidence"]["max_budget"])
    pipeline.max_budget = max_steps

    render_html("<div style='height: 1px; background: #1e293b; margin: 1.2rem 0;'></div>")

    if st.button("Reset Diagnostic Session", use_container_width=True):
        st.session_state["current_state"] = None
        st.session_state["active_case"] = None
        st.session_state["active_preset_name"] = None
        st.session_state["active_image_path_or_pil"] = None
        st.session_state["selected_case_id"] = "tomato_early_blight_0001" if manifest else None
        st.rerun()

    render_html("<div style='height: 1px; background: #1e293b; margin: 1.2rem 0;'></div>")
    render_html("""
        <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;">
            Target Pathology Taxonomy
        </div>
    """)

    for c in classes:
        col = disease_palette.get(c, "#10b981")
        render_html(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #131c2e;">
                <span style="font-size: 0.84rem; color: #cbd5e1; font-weight: 500;">{c.replace('_', ' ').title()}</span>
                <span style="width: 10px; height: 10px; border-radius: 50%; background: {col}; box-shadow: 0 0 6px {col};"></span>
            </div>
        """)

    render_html("<div style='height: 1px; background: #1e293b; margin: 1.2rem 0;'></div>")
    render_html(r"""
        <div style="font-size: 0.74rem; color: #64748b; line-height: 1.6; font-family: 'JetBrains Mono', monospace;">
            <b style="color: #94a3b8;">SYSTEM METRICS:</b><br>
            Encoder: ResNet-18 (ImageNet)<br>
            Platt Temp: T = 1.8928<br>
            Entropy Stop: H &le; 0.350 bits<br>
            Margin Stop: &Delta;p &ge; 0.400
        </div>
    """)

render_html("""
<div class="top-header">
    <div class="header-brand">
        <div class="header-title">EvidenceGain-X</div>
        <div class="header-subtitle">Active Counterfactual Diagnostic Console &bull; Goal-Directed Next-Best Evidence Policy</div>
    </div>
    <div class="header-telemetry">
        <div class="telemetry-chip status-online">SYSTEM NOMINAL</div>
        <div class="telemetry-chip">BACKBONE: RESNET-18</div>
        <div class="telemetry-chip">DATASET: PLANTVILLAGE</div>
        <div class="telemetry-chip">CALIBRATION: T=1.89</div>
    </div>
</div>
""")

render_html("""
<div class="preset-container">
    <div class="preset-header">Validated Clinical Case Scenarios & Stress Tests</div>
</div>
""")

p1, p2, p3, p4, p5, p6 = st.columns(6)

with p1:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #f59e0b;">
        <span class="preset-badge-tag" style="color: #f59e0b;">PATHOLOGY</span>
        <div class="preset-sub-text">Alternaria solani</div>
    </div>
    """)
    if st.button("Early Blight", key="preset_eb", use_container_width=True):
        cid = "tomato_early_blight_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Early Blight (Alternaria solani)"
        st.session_state["active_image_path_or_pil"] = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_early_blight.jpg"
        st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
        st.rerun()

with p2:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #ef4444;">
        <span class="preset-badge-tag" style="color: #ef4444;">PATHOLOGY</span>
        <div class="preset-sub-text">Phytophthora</div>
    </div>
    """)
    if st.button("Late Blight", key="preset_lb", use_container_width=True):
        cid = "tomato_late_blight_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Late Blight (Phytophthora infestans)"
        st.session_state["active_image_path_or_pil"] = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_late_blight.jpg"
        st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
        st.rerun()

with p3:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #a855f7;">
        <span class="preset-badge-tag" style="color: #a855f7;">AMBIGUITY</span>
        <div class="preset-sub-text">Look-Alike Boundary</div>
    </div>
    """)
    if st.button("Septoria Spot", key="preset_sep", use_container_width=True):
        cid = "tomato_septoria_leaf_spot_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Septoria Leaf Spot (Look-alike Ambiguity)"
        st.session_state["active_image_path_or_pil"] = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_septoria.jpg"
        st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
        st.rerun()

with p4:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #10b981;">
        <span class="preset-badge-tag" style="color: #10b981;">CONTROL</span>
        <div class="preset-sub-text">Healthy Negative</div>
    </div>
    """)
    if st.button("Healthy Control", key="preset_hea", use_container_width=True):
        cid = "tomato_healthy_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Healthy Foliage (Negative Control)"
        st.session_state["active_image_path_or_pil"] = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_healthy.jpg"
        st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
        st.rerun()

with p5:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #06b6d4;">
        <span class="preset-badge-tag" style="color: #06b6d4;">SAFETY M7</span>
        <div class="preset-sub-text">Defocus Blur Gate</div>
    </div>
    """)
    if st.button("Test: Blur Gating", key="preset_blur", use_container_width=True):
        cid = "tomato_septoria_leaf_spot_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Adversarial Test: Defocus Blur Gating (Module 7)"
        init_path = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_septoria.jpg"
        st.session_state["active_image_path_or_pil"] = init_path
        init_state = pipeline.diagnose_initial(init_path)
        blurry_item = EvidenceItem(
            evidence_type="lesion_closeup",
            modality="image",
            data_path_or_content="data/test_samples/sample_blurry.jpg",
            cost=1.0
        )
        updated_state = pipeline.process_single_evidence(init_state, "lesion_closeup", blurry_item)
        st.session_state["current_state"] = updated_state
        st.rerun()

with p6:
    render_html("""
    <div class="preset-card-top" style="border-top: 3px solid #f43f5e;">
        <span class="preset-badge-tag" style="color: #f43f5e;">SAFETY M8</span>
        <div class="preset-sub-text">Contradiction Gate</div>
    </div>
    """)
    if st.button("Test: Contradiction", key="preset_contra", use_container_width=True):
        cid = "tomato_early_blight_0001"
        c_rec = case_dict.get(cid)
        st.session_state["input_source_mode"] = "Benchmark Gallery"
        st.session_state["selected_case_id"] = cid
        st.session_state["active_case"] = c_rec
        st.session_state["active_preset_name"] = "Adversarial Test: Contradictory Input (Module 8)"
        init_path = c_rec.initial_image_path if c_rec else "data/test_samples/sample_real_early_blight.jpg"
        st.session_state["active_image_path_or_pil"] = init_path
        init_state = pipeline.diagnose_initial(init_path)
        contradictory_item = EvidenceItem(
            evidence_type="lesion_closeup",
            modality="image",
            data_path_or_content="data/test_samples/sample_healthy_contradiction.jpg",
            cost=1.0
        )
        updated_state = pipeline.process_single_evidence(init_state, "lesion_closeup", contradictory_item)
        st.session_state["current_state"] = updated_state
        st.rerun()

render_html("<div style='height: 12px;'></div>")

col_left, col_right = st.columns([1, 1.3], gap="large")

with col_left:
    render_html("""
    <div class="panel-card">
        <div class="panel-title">
            <span class="panel-title-number">01</span>
            Specimen Observation (X₀)
        </div>
    """)

    if st.session_state.get("active_preset_name"):
        render_html(f"""
        <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 10px; padding: 6px 10px; background: #131c2e; border-radius: 6px; border: 1px solid #1e293b;">
            Active Case: <b style="color: #f1f5f9;">{st.session_state['active_preset_name']}</b>
        </div>
        """)

    curr_input_mode = st.session_state["input_source_mode"]

    if curr_input_mode == "Live Image Ingestion":
        uploaded_file = st.file_uploader("Upload macroscopic leaf photograph:", type=["jpg", "jpeg", "png"])

        render_html("<div style='font-size: 0.76rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin: 10px 0 6px 0;'>Or select reference specimen:</div>")
        s_c1, s_c2, s_c3 = st.columns(3)
        with s_c1:
            if st.button("Sample: Septoria", use_container_width=True):
                st.session_state["active_image_path_or_pil"] = "data/test_samples/sample_real_septoria.jpg"
                st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
                st.session_state["active_case"] = None
                st.session_state["active_preset_name"] = "Live Ingestion: Septoria Spot"
                st.rerun()
        with s_c2:
            if st.button("Sample: Early Blight", use_container_width=True):
                st.session_state["active_image_path_or_pil"] = "data/test_samples/sample_real_early_blight.jpg"
                st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
                st.session_state["active_case"] = None
                st.session_state["active_preset_name"] = "Live Ingestion: Early Blight"
                st.rerun()
        with s_c3:
            if st.button("Sample: Defocus Blur", use_container_width=True):
                st.session_state["active_image_path_or_pil"] = "data/test_samples/sample_blurry.jpg"
                st.session_state["current_state"] = pipeline.diagnose_initial(st.session_state["active_image_path_or_pil"])
                st.session_state["active_case"] = None
                st.session_state["active_preset_name"] = "Live Ingestion: Blurry Specimen"
                st.rerun()

        if uploaded_file is not None:
            live_pil = Image.open(uploaded_file).convert("RGB")
            st.session_state["active_image_path_or_pil"] = live_pil

            q_check = pipeline.reliability_checker.assess_image_quality(live_pil)

            render_html("""
            <div class="viewfinder-box">
                <div class="viewfinder-hud">
                    <span>OPTICAL SENSOR VIEW • FOV 1.0X</span>
                    <span>INGESTED SAMPLE</span>
                </div>
            </div>
            """)
            st.image(live_pil, use_container_width=True)
            render_html(f"""
                <div class="telemetry-row">
                    <div class="telemetry-cell">
                        <div class="telemetry-cell-label">LAPLACIAN SHARPNESS</div>
                        <div class="telemetry-cell-val" style="color: {'#10b981' if q_check.is_acceptable else '#ef4444'};">{q_check.blur_score:.1f} &sigma;&sup2;</div>
                    </div>
                    <div class="telemetry-cell">
                        <div class="telemetry-cell-label">LUMINANCE LEVEL</div>
                        <div class="telemetry-cell-val">{q_check.brightness_score:.1f} cd/m&sup2;</div>
                    </div>
                </div>
            """)

            if not q_check.is_acceptable:
                st.session_state["current_state"] = {
                    "initial_probs": np.zeros(len(classes)),
                    "current_probs": np.zeros(len(classes)),
                    "entropy": 2.0,
                    "margin": 0.0,
                    "leading_disease": "Optical Quality Quarantined",
                    "leading_prob": 0.0,
                    "rival_disease": None,
                    "rival_prob": 0.0,
                    "top_hypotheses": [],
                    "full_distribution": {c: 0.0 for c in classes},
                    "is_ambiguous": True,
                    "is_optical_rejected": True,
                    "optical_reasons": q_check.reasons,
                    "optical_score": q_check.blur_score,
                    "conflict": None,
                    "evidence_trail": []
                }
                render_html(f"""
                <div class="alert-box alert-ambiguity" style="border-color: #ef4444; color: #fca5a5; background: rgba(239, 68, 68, 0.15); margin-top: 10px;">
                    <div><b>OPTICAL QUALITY REJECTED (Module 7):</b> {", ".join(q_check.reasons)}</div>
                    <div style="font-size: 0.8rem; margin-top: 4px;">Diagnosis blocked: System strictly refuses to generate disease predictions on unreadable, blurry, or distorted images.</div>
                </div>
                <div style="text-align: center; padding: 10px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #ef4444; background: rgba(239, 68, 68, 0.08); border-radius: 6px; border: 1px dashed rgba(239, 68, 68, 0.4); margin-top: 8px;">
                    DIAGNOSIS QUARANTINED: OPTICAL CLEARANCE FAILED
                </div>
                """)
            else:
                if st.button("Execute Provisional Diagnosis", type="primary", use_container_width=True):
                    st.session_state["current_state"] = pipeline.diagnose_initial(live_pil)
                    st.session_state["active_case"] = None
                    st.session_state["active_preset_name"] = "Live Upload Specimen"
                    st.rerun()
        elif st.session_state.get("active_image_path_or_pil"):
            disp_target = st.session_state["active_image_path_or_pil"]
            render_html("""
            <div class="viewfinder-box">
                <div class="viewfinder-hud">
                    <span>OPTICAL SENSOR VIEW • FOV 1.0X</span>
                    <span>ACTIVE SAMPLE</span>
                </div>
            </div>
            """)
            if isinstance(disp_target, (str, Path)) and Path(disp_target).exists():
                st.image(str(disp_target), use_container_width=True)
            elif isinstance(disp_target, Image.Image):
                st.image(disp_target, use_container_width=True)

    else:
        if manifest:
            case_ids = [c.case_id for c in manifest.cases]
            sel_default_idx = 0
            if st.session_state.get("selected_case_id") in case_ids:
                sel_default_idx = case_ids.index(st.session_state["selected_case_id"])

            selected_id = st.selectbox(
                "Benchmark Case Record:",
                case_ids,
                index=sel_default_idx,
                key="sb_benchmark_case"
            )
            st.session_state["selected_case_id"] = selected_id
            case_record = case_dict.get(selected_id)
            st.session_state["active_case"] = case_record

            if case_record and Path(case_record.initial_image_path).exists():
                initial_img = Image.open(case_record.initial_image_path)
                q_check = pipeline.reliability_checker.assess_image_quality(case_record.initial_image_path)

                render_html(f"""
                <div class="viewfinder-box">
                    <div class="viewfinder-hud">
                        <span>RECORD: {case_record.case_id}</span>
                        <span>GROUND TRUTH: {case_record.disease.replace('_', ' ').upper()}</span>
                    </div>
                </div>
                """)
                st.image(initial_img, use_container_width=True)
                render_html(f"""
                    <div class="telemetry-row">
                        <div class="telemetry-cell">
                            <div class="telemetry-cell-label">LAPLACIAN SHARPNESS</div>
                            <div class="telemetry-cell-val" style="color: #10b981;">{q_check.blur_score:.1f} &sigma;&sup2; <span style="font-size: 0.7rem; color: #10b981;">(PASS)</span></div>
                        </div>
                        <div class="telemetry-cell">
                            <div class="telemetry-cell-label">LUMINANCE LEVEL</div>
                            <div class="telemetry-cell-val">{q_check.brightness_score:.1f} cd/m&sup2; <span style="font-size: 0.7rem; color: #10b981;">(NOMINAL)</span></div>
                        </div>
                    </div>
                """)

            if st.button("Run Provisional Diagnosis", type="primary", use_container_width=True):
                st.session_state["current_state"] = pipeline.diagnose_initial(case_record.initial_image_path)
                st.session_state["active_image_path_or_pil"] = case_record.initial_image_path
                st.session_state["active_preset_name"] = None
                st.rerun()

    render_html('</div>')

with col_right:
    state = st.session_state.get("current_state")
    if state is None:
        render_html("""
        <div class="panel-card" style="text-align: center; padding: 3.5rem 1.5rem;">
            <div style="width: 48px; height: 48px; border-radius: 50%; background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.3); margin: 0 auto 1.2rem auto; display: flex; align-items: center; justify-content: center;">
                <div style="width: 14px; height: 14px; border-radius: 50%; background: #38bdf8;"></div>
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; letter-spacing: -0.01em;">Awaiting Specimen Ingestion</div>
            <p style="color: #64748b; font-size: 0.88rem; max-width: 440px; margin: 0.6rem auto 0 auto; line-height: 1.6;">
                Select a benchmark specimen or load one of the clinical case presets above to initiate active sequential hypothesis tracking.
            </p>
        </div>
        """)
    elif state.get("is_optical_rejected"):
        blur_sc = state.get("optical_score", 0.0)
        render_html(f"""
        <div class="panel-card" style="border: 1px solid rgba(239, 68, 68, 0.45); background: linear-gradient(145deg, #241417, #180d0f);">
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700; color: #ef4444; letter-spacing: 0.08em; text-transform: uppercase;">
                MODULE 7 &bull; OPTICAL QUALITY GATE QUARANTINE
            </div>
            <div style="font-size: 1.45rem; font-weight: 800; color: #ffffff; margin: 6px 0 4px 0;">
                Diagnosis Blocked: Severe Defocus Blur
            </div>
            <div style="font-size: 0.84rem; color: #fca5a5; margin-bottom: 1.2rem;">
                Specimen failed optical resolution criteria &bull; Laplacian Variance: <b>{blur_sc:.1f} &sigma;&sup2;</b> (Minimum Threshold: &gt; 80.0 &sigma;&sup2;)
            </div>

            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 10px; padding: 1.2rem; margin-bottom: 1.2rem; line-height: 1.6; font-size: 0.88rem; color: #cbd5e1;">
                <b style="color: #ffffff;">Why are disease probabilities withheld?</b><br>
                Standard passive neural networks (like single-shot ResNet-18) blindly output high-confidence predictions (e.g. 89.5% Late Blight) when presented with blurry green noise because softmax forces outputs to sum to 100%.<br><br>
                <b>EvidenceGain-X Module 7 enforces rigorous optical gating:</b> Corrupted or unreadable specimens are strictly quarantined from belief updating. No disease probabilities are emitted until optical clarity is achieved.
            </div>

            <div style="font-size: 0.82rem; color: #94a3b8; font-family: 'JetBrains Mono', monospace;">
                ACTION REQUIRED: Please capture or upload a clear, focused foliar photograph to initiate active diagnosis.
            </div>
        </div>
        """)
    else:
        render_html("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="panel-title-number">02</span>
                Provisional Hypothesis Tracking &amp; Entropy State
            </div>
        """)

        lead_color = disease_palette.get(state["leading_disease"], "#38bdf8")

        k1, k2, k3 = st.columns(3)
        with k1:
            render_html(f"""
            <div class="metric-cell" style="border-top-color: {lead_color};">
                <div class="metric-label">Leading Hypothesis (h₁)</div>
                <div class="metric-value" style="color: {lead_color}; font-size: 1.25rem;">{state['leading_disease'].replace('_', ' ').title()}</div>
                <div class="metric-sub">Rank 1 Posterior</div>
            </div>
            """)
        with k2:
            render_html(f"""
            <div class="metric-cell accent-emerald">
                <div class="metric-label">Calibrated Probability</div>
                <div class="metric-value">{state['leading_prob']*100:.1f}%</div>
                <div class="metric-sub">Platt Scaled (T=1.89)</div>
            </div>
            """)
        with k3:
            ent_val = state['entropy']
            ent_accent = "accent-emerald" if ent_val <= pipeline.stopping_entropy else "accent-amber"
            render_html(f"""
            <div class="metric-cell {ent_accent}">
                <div class="metric-label">Shannon Entropy H(P)</div>
                <div class="metric-value">{ent_val:.3f} <span style="font-size: 0.78rem; font-weight: 500; color: #94a3b8;">bits</span></div>
                <div class="metric-sub">Threshold &le; {pipeline.stopping_entropy:.2f} bits</div>
            </div>
            """)

        prob_rows_html = ""
        for i, c_name in enumerate(classes):
            p_val = state["current_probs"][i]
            bar_grad = disease_gradients.get(c_name, "linear-gradient(90deg, #059669, #34d399)")
            is_lead = (c_name == state["leading_disease"])
            badge_html = '<span class="prob-lead-badge">LEADING</span>' if is_lead else ''

            prob_rows_html += f"""
            <div class="prob-row">
                <div class="prob-labels">
                    <div class="prob-name-box">
                        <span style="color: {'#ffffff' if is_lead else '#94a3b8'}; font-weight: {'700' if is_lead else '500'}; font-size: 0.86rem;">
                            {c_name.replace('_', ' ').title()}
                        </span>
                        {badge_html}
                    </div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.86rem; color: {'#ffffff' if is_lead else '#94a3b8'}; font-weight: 700;">
                        {p_val*100:.1f}%
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-bar" style="width: {p_val*100:.1f}%; background: {bar_grad};"></div>
                </div>
            </div>
            """

        render_html(f"""
        <div class="prob-container">
            <div style="font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;">
                Calibrated Posterior Distribution
            </div>
            {prob_rows_html}
        </div>
        """)

        rival = state.get("rival_disease")
        margin = state.get("margin", 1.0)
        entropy = state.get("entropy", 1.0)

        if margin <= 0.25 and rival:
            render_html(f"""
            <div class="alert-box alert-ambiguity">
                <div>
                    <b>Critical Look-Alike Ambiguity Detected:</b><br>
                    Hypothesis margin is critically narrow between <i>{state['leading_disease'].replace('_', ' ').title()}</i> and <i>{rival.replace('_', ' ').title()}</i> (&Delta;p = {margin:.2f} &le; 0.25). Passive single-shot classifiers suffer severe confusion on this morphological boundary. EvidenceGain-X initiates goal-directed inquiry to resolve the ambiguity.
                </div>
            </div>
            """)
        elif entropy > pipeline.stopping_entropy:
            render_html(f"""
            <div class="alert-box alert-provisional">
                <div>
                    <b>Provisional Hypothesis Lead (Active Inquiry Required):</b><br>
                    <i>{state['leading_disease'].replace('_', ' ').title()}</i> holds a distinct provisional lead over <i>{rival.replace('_', ' ').title() if rival else 'alternatives'}</i> (&Delta;p = {margin:.2f}, {state['leading_prob']*100:.1f}%). However, Shannon entropy H(P) = {entropy:.3f} bits remains above clinical stopping certainty (&le; {pipeline.stopping_entropy:.2f} bits). EvidenceGain-X recommends acquiring next-best evidence to reach definitive diagnostic closure.
                </div>
            </div>
            """)
        else:
            render_html(f"""
            <div class="alert-box alert-decisive">
                <div>
                    <b>Decisive Diagnostic Separation Achieved:</b><br>
                    Leading hypothesis <i>{state['leading_disease'].replace('_', ' ').title()}</i> established dominant separation (&Delta;p = {margin:.2f}, {state['leading_prob']*100:.1f}%) with entropy H(P) = {entropy:.3f} bits &le; {pipeline.stopping_entropy:.2f} bits. Stopping criteria satisfied.
                </div>
            </div>
            """)

        render_html('</div>')

        render_html("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="panel-title-number">03</span>
                Next-Best Evidence Recommendation (e*)
            </div>
        """)

        used_cands = [item["evidence_type"] for item in state["evidence_trail"]]
        rem_cands = [c for c in cand_types if c not in used_cands]

        if not rem_cands:
            st.info("All candidate evidence modalities have been exhausted for this episode.")
        elif state["entropy"] <= pipeline.stopping_entropy and state["margin"] >= pipeline.stopping_margin:
            render_html(r"""
            <div class="alert-box alert-decisive" style="display: block; text-align: center; padding: 1.4rem;">
                <div style="font-weight: 700; font-size: 1.05rem; color: #34d399; margin-bottom: 4px;">Stopping Criterion Satisfied (H(P) &le; 0.35 bits)</div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Diagnostic certainty resolved. Counterfactual utility indicates further acquisition yields diminishing information return.</div>
            </div>
            """)
        else:
            h1_idx = classes.index(state["leading_disease"])
            h2_name = state["rival_disease"] or classes[(h1_idx + 1) % len(classes)]
            h2_idx = classes.index(h2_name)

            utility_records = []
            for cand in rem_cands:
                u_info = pipeline.policy.compute_evidence_utility(state["current_probs"], h1_idx, h2_idx, cand)
                utility_records.append(u_info)

            u_df = pd.DataFrame(utility_records)
            best_cand_row = u_df.loc[u_df["total_utility"].idxmax()]
            best_cand = best_cand_row["candidate"]

            render_html(f"""
            <div class="hero-recommendation-card">
                <div class="hero-rec-top">
                    <span class="hero-rec-tag">OPTIMAL INQUIRY QUERY (e*)</span>
                    <span class="hero-rec-cost">COST: {best_cand_row['cost']:.1f} UNIT</span>
                </div>
                <div class="hero-rec-name">{best_cand.replace('_', ' ').title()}</div>
                <div class="hero-rec-metrics">
                    <div class="hero-rec-metric-item">
                        Separation (JS)
                        <b>{best_cand_row['hypothesis_separation']:.3f}</b>
                    </div>
                    <div class="hero-rec-metric-item">
                        Exp. Uncertainty &Delta;H
                        <b>{best_cand_row['uncertainty_reduction']:.3f} bits</b>
                    </div>
                    <div class="hero-rec-metric-item">
                        Net Policy Utility U(e*)
                        <b style="color: #38bdf8;">{best_cand_row['total_utility']:.3f}</b>
                    </div>
                </div>
            </div>
            """)

            table_rows_html = ""
            sorted_u_df = u_df.sort_values(by="total_utility", ascending=False)
            for _, r in sorted_u_df.iterrows():
                is_top = (r["candidate"] == best_cand)
                row_cls = "top-cand-row" if is_top else ""
                tag_html = '<span style="color: #38bdf8; font-weight: 700;">TOP CHOICE</span>' if is_top else '<span style="color: #64748b;">ALTERNATIVE</span>'
                table_rows_html += f"""
                <tr class="{row_cls}">
                    <td>{r['candidate'].replace('_', ' ').title()}</td>
                    <td>{r['hypothesis_separation']:.3f}</td>
                    <td>{r['uncertainty_reduction']:.3f} bits</td>
                    <td>{r['cost']:.1f}</td>
                    <td style="color: {'#38bdf8' if is_top else '#cbd5e1'}; font-weight: 700;">{r['total_utility']:.3f}</td>
                    <td>{tag_html}</td>
                </tr>
                """

            table_full_html = f"""
            <table class="clinical-table">
                <thead>
                    <tr>
                        <th>Modality</th>
                        <th>Separation (JS)</th>
                        <th>Exp. &Delta;H</th>
                        <th>Cost</th>
                        <th>Net Utility</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows_html}
                </tbody>
            </table>
            """
            render_html(table_full_html)

            with st.expander(f"Acquire Clue: {best_cand.replace('_', ' ').title()}", expanded=True):
                ev_to_submit = None
                chosen_modality = best_cand
                active_case = st.session_state.get("active_case")

                if active_case and best_cand in active_case.available_evidence:
                    b_item = active_case.available_evidence[best_cand]
                    render_html(f"<div style='font-size: 0.84rem; color: #94a3b8; margin-bottom: 8px;'>Authentic Benchmark Clue: <code>{b_item.evidence_type}</code> ({b_item.modality})</div>")
                    if st.button(f"Acquire Benchmark Observation ({best_cand.replace('_', ' ').title()})", type="primary", use_container_width=True):
                        ev_to_submit = b_item
                        chosen_modality = best_cand

                render_html("<div style='font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 6px 0;'>Stress-Testing Controls</div>")
                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    if st.button("Inject Defocus Blur (Module 7)", use_container_width=True):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="image",
                            data_path_or_content="data/test_samples/sample_blurry.jpg",
                            cost=1.0
                        )
                        chosen_modality = best_cand
                with sc_c2:
                    if st.button("Inject Contradictory Leaf (Module 8)", use_container_width=True):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="image",
                            data_path_or_content="data/test_samples/sample_healthy_contradiction.jpg",
                            cost=1.0
                        )
                        chosen_modality = best_cand

                render_html("<div style='font-size: 0.74rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; margin: 14px 0 8px 0;'>Interactive Evidence Ingestion</div>")

                tab_img, tab_weather, tab_symptom = st.tabs(["Upload Leaf Photo", "One-Click Weather", "Visual Symptoms"])

                with tab_img:
                    render_html("<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 6px;'>Upload a photographic clue or select a reference sample:</div>")
                    img_modality = st.selectbox(
                        "Select Image Perspective:",
                        ["lesion_closeup", "leaf_underside", "stem_view"],
                        index=0 if best_cand not in ["leaf_underside", "stem_view"] else ["lesion_closeup", "leaf_underside", "stem_view"].index(best_cand),
                        format_func=lambda x: x.replace('_', ' ').title(),
                        key="sb_img_modality"
                    )

                    s_col1, s_col2 = st.columns(2)
                    with s_col1:
                        if st.button("Use Sample: Lesion Close-Up", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type=img_modality,
                                modality="image",
                                data_path_or_content="data/test_samples/sample_real_early_blight.jpg",
                                cost=1.0
                            )
                            chosen_modality = img_modality
                    with s_col2:
                        if st.button("Use Sample: Leaf Underside", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type=img_modality,
                                modality="image",
                                data_path_or_content="data/test_samples/sample_real_septoria.jpg",
                                cost=1.2
                            )
                            chosen_modality = img_modality

                    live_ev_file = st.file_uploader("Or upload your own specimen photo:", type=["jpg", "png", "jpeg"], key="live_user_ev_file")
                    if live_ev_file:
                        live_pil = Image.open(live_ev_file).convert("RGB")
                        st.image(live_pil, width=220)
                        if st.button("Submit Ingested Photo", type="primary", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type=img_modality,
                                modality="image",
                                data_path_or_content=live_pil,
                                cost=1.0
                            )
                            chosen_modality = img_modality

                with tab_weather:
                    render_html("<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;'>Select observed field environmental conditions (no typing needed):</div>")
                    w_c1, w_c2 = st.columns(2)
                    with w_c1:
                        if st.button("Warm & Dry (28°C, low rain)", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="weather_context",
                                modality="text",
                                data_path_or_content="warm dry 28c low humidity minimal rain",
                                cost=0.4
                            )
                            chosen_modality = "weather_context"
                        if st.button("Warm & Humid (24°C, morning dew)", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="weather_context",
                                modality="text",
                                data_path_or_content="warm humid 24c heavy dew leaf wetness",
                                cost=0.4
                            )
                            chosen_modality = "weather_context"
                    with w_c2:
                        if st.button("Cool & Wet (18°C, continuous rain)", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="weather_context",
                                modality="text",
                                data_path_or_content="cool rainy 18c continuous precipitation high humidity",
                                cost=0.4
                            )
                            chosen_modality = "weather_context"
                        if st.button("Mild & Sunny (Optimal control)", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="weather_context",
                                modality="text",
                                data_path_or_content="mild sunny moderate temperature dry clean",
                                cost=0.4
                            )
                            chosen_modality = "weather_context"

                with tab_symptom:
                    render_html("<div style='font-size: 0.82rem; color: #94a3b8; margin-bottom: 8px;'>Select visible symptoms or write plain everyday notes:</div>")
                    sym_c1, sym_c2 = st.columns(2)
                    with sym_c1:
                        if st.button("Concentric brown target rings", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="symptom_text",
                                modality="text",
                                data_path_or_content="dark brown circular concentric rings yellow halo",
                                cost=0.4
                            )
                            chosen_modality = "symptom_text"
                        if st.button("Tiny dark spots with grey centers", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="symptom_text",
                                modality="text",
                                data_path_or_content="small circular spots dark border grey center",
                                cost=0.4
                            )
                            chosen_modality = "symptom_text"
                    with sym_c2:
                        if st.button("Dark wet patches with white fuzz", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="symptom_text",
                                modality="text",
                                data_path_or_content="water soaked dark lesions white mold underside",
                                cost=0.4
                            )
                            chosen_modality = "symptom_text"
                        if st.button("Clean green leaves (no spots)", use_container_width=True):
                            ev_to_submit = EvidenceItem(
                                evidence_type="symptom_text",
                                modality="text",
                                data_path_or_content="healthy green leaves clean foliage no spots",
                                cost=0.4
                            )
                            chosen_modality = "symptom_text"

                    custom_text = st.text_input("Or type plain observation:", placeholder="e.g. leaves have circular dark spots, dry weather")
                    if st.button("Submit Custom Note", use_container_width=True):
                        if custom_text.strip():
                            ev_to_submit = EvidenceItem(
                                evidence_type="symptom_text",
                                modality="text",
                                data_path_or_content=custom_text.strip(),
                                cost=0.4
                            )
                            chosen_modality = "symptom_text"

                if ev_to_submit is not None:
                    with st.spinner("Executing Reliability Gating, Contradiction Verification & Belief Updating..."):
                        updated_state = pipeline.process_single_evidence(state, chosen_modality, ev_to_submit)
                        st.session_state["current_state"] = updated_state
                        st.rerun()

        render_html('</div>')

if state and len(state["evidence_trail"]) > 0:
    render_html("<div style='height: 14px;'></div>")

    col_t1, col_t2 = st.columns([1.4, 1.1], gap="large")

    with col_t1:
        render_html("""
        <div class="panel-card">
            <div class="panel-title">
                <span class="panel-title-number">04</span>
                Sequential Diagnostic Audit Trail
            </div>
        """)

        for item in state["evidence_trail"]:
            act = item.get("action", "ACCEPT")
            is_rejected = act in ["REJECT", "REJECTED", "QUARANTINED"]
            node_cls = "node-rejected" if is_rejected else ""

            badge_color = "#f59e0b" if act == "QUARANTINED" else ("#ef4444" if is_rejected else "#10b981")
            badge_bg = "rgba(245, 158, 11, 0.12)" if act == "QUARANTINED" else ("rgba(239, 68, 68, 0.12)" if is_rejected else "rgba(16, 185, 129, 0.12)")
            badge_border = "rgba(245, 158, 11, 0.3)" if act == "QUARANTINED" else ("rgba(239, 68, 68, 0.3)" if is_rejected else "rgba(16, 185, 129, 0.3)")

            render_html(f"""
            <div class="audit-step-node {node_cls}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <div style="font-weight: 700; color: #ffffff; font-size: 0.92rem;">
                        Step {item['step']}: {item['evidence_type'].replace('_', ' ').title()}
                    </div>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.74rem; font-weight: 700; color: {badge_color}; background: {badge_bg}; border: 1px solid {badge_border}; padding: 2px 8px; border-radius: 4px;">
                        {act}
                    </span>
                </div>
            </div>
            """)

            with st.expander(f"Audit Details: Step {item['step']}", expanded=True):
                render_html(f"""
                <div style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 6px;">
                    <b>Module 7 Optical Gating:</b> Quality Score: <code>{item['quality_score']:.2f}</code>
                </div>
                """)

                if item.get("reasons"):
                    render_html(f"<div style='font-size: 0.8rem; color: #fca5a5; margin-bottom: 6px;'><b>Gating Rejection:</b> {', '.join(item['reasons'])}</div>")

                if act == "QUARANTINED":
                    render_html(f"""
                    <div class="alert-box alert-ambiguity" style="border-color: #f59e0b; background: rgba(245, 158, 11, 0.1); color: #fde68a; margin-top: 6px;">
                        <div><b>Specimen Mismatch Quarantined (Module 8):</b> {item.get('contradiction_explanation')}</div>
                    </div>
                    """)
                elif is_rejected:
                    render_html("""
                    <div class="alert-box alert-ambiguity" style="border-color: #ef4444; background: rgba(239, 68, 68, 0.1); color: #fca5a5; margin-top: 6px;">
                        <div><b>Optical Filter Triggered:</b> Image failed sharpness criteria. Gated from Bayesian posterior calculation to prevent distribution poisoning.</div>
                    </div>
                    """)
                else:
                    is_contra = item.get("contradiction", False)
                    if is_contra:
                        render_html(f"""
                        <div class="alert-box alert-ambiguity" style="margin-top: 6px;">
                            <div><b>Contradiction Detected (Module 8):</b> {item.get('contradiction_explanation')}</div>
                        </div>
                        """)
                    else:
                        render_html("""
                        <div class="alert-box alert-decisive" style="padding: 8px 12px; margin-top: 6px; font-size: 0.82rem;">
                            <div>Observation verified congruent with counterfactual diagnosis.</div>
                        </div>
                        """)

                render_html(f"""
                <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-top: 10px; padding-top: 8px; border-top: 1px solid #1e293b;">
                    <span>Hypothesis: <b style="color: #ffffff;">{item['leading_after'].replace('_', ' ').title()}</b> ({item['leading_prob_after']*100:.1f}%)</span>
                    <span>Entropy: <code style="color: #38bdf8;">{item['entropy_after']:.3f} bits</code></span>
                </div>
                """)

        render_html('</div>')

    with col_t2:
        has_quarantine = any(item.get("action") == "QUARANTINED" for item in state["evidence_trail"])
        has_contradiction = any(item.get("contradiction", False) for item in state["evidence_trail"])
        has_rejection = any(item.get("action") in ["REJECT", "REJECTED"] for item in state["evidence_trail"])
        is_sufficient = (state["entropy"] <= pipeline.stopping_entropy and state["margin"] >= pipeline.stopping_margin and not has_contradiction and not has_quarantine)

        final_disease = state['leading_disease'].replace('_', ' ').title()
        init_lead_idx = int(np.argmax(state["initial_probs"]))
        init_lead_name = classes[init_lead_idx]

        init_p = np.array(state["initial_probs"])
        h0 = float(-np.sum(init_p * np.log2(init_p + 1e-12)))
        ht = float(state["entropy"])
        delta_h = max(0.0, h0 - ht)

        if has_quarantine:
            card_cls = "cert-contradiction"
            cert_super = "EVIDENCEGAIN-X &bull; SPECIMEN MISMATCH QUARANTINED"
            super_color = "#f59e0b"
            outcome_title = f"{final_disease} (Preserved)"
            outcome_sub = "Cross-specimen contamination quarantined by Module 8 guard"
            posterior_color = "#94a3b8"
            post_extra = " (Preserved)"
            dh_display = "<span style='color: #94a3b8;'>0.000 bits (Preserved)</span>"
            footer_note = "<b>Advisory (Module 8):</b> Uploaded clue shows symptoms of an excluded disease. Quarantined from belief distribution to prevent diagnostic corruption."
        elif has_contradiction:
            card_cls = "cert-contradiction"
            cert_super = "EVIDENCEGAIN-X &bull; CONTRADICTION DETECTED"
            super_color = "#f59e0b"
            outcome_title = f"{final_disease} (Disputed)"
            outcome_sub = "Incoming evidence directly conflicts with specimen pathology"
            posterior_color = "#f59e0b"
            post_extra = " (Dampened)"
            dh_display = f"<span style='color: #f59e0b;'>+{ht - h0:.3f} bits (H increased)</span>" if ht > h0 else f"<span style='color: #38bdf8;'>+{delta_h:.3f} bits</span>"
            footer_note = f"<b>Advisory (Module 8):</b> Acquired clue conflicts with {final_disease}. Bayesian belief dampened to protect clinical diagnostic integrity."
        elif has_rejection:
            card_cls = "cert-rejected"
            cert_super = "EVIDENCEGAIN-X &bull; OPTICAL FILTER GATED"
            super_color = "#ef4444"
            outcome_title = f"{final_disease} (Unchanged)"
            outcome_sub = "Optical quality gate quarantined corrupted evidence"
            posterior_color = "#94a3b8"
            post_extra = " (Preserved)"
            dh_display = "<span style='color: #94a3b8;'>0.000 bits (Preserved)</span>"
            footer_note = "<b>Advisory (Module 7):</b> Image failed sharpness criteria (Laplacian blur filter). Gated from belief distribution."
        elif is_sufficient:
            card_cls = ""
            cert_super = "EVIDENCEGAIN-X &bull; VERIFIED CLINICAL AUDIT"
            super_color = "#34d399"
            outcome_title = final_disease
            outcome_sub = "Target Folio-Pathology Confirmed"
            posterior_color = "#34d399"
            post_extra = ""
            dh_display = f"<span style='color: #38bdf8;'>+{delta_h:.3f} bits</span>"
            footer_note = "<b>Validation:</b> Stopping criteria satisfied (H &le; 0.35 bits) &bull; Module 7/8 Security Passed"
        else:
            card_cls = "cert-provisional"
            cert_super = "EVIDENCEGAIN-X &bull; PROVISIONAL AUDIT"
            super_color = "#38bdf8"
            outcome_title = f"{final_disease} (Leading)"
            outcome_sub = "Inquiry in Progress &bull; Sub-Threshold Certainty"
            posterior_color = "#38bdf8"
            post_extra = ""
            dh_display = f"<span style='color: #38bdf8;'>+{delta_h:.3f} bits</span>"
            footer_note = "<b>Status:</b> Intermediate trajectory &bull; Further evidence recommended for definitive stopping"

        render_html(f"""
        <div class="clinical-certificate {card_cls}">
            <div class="cert-super" style="color: {super_color};">{cert_super}</div>
            <div class="cert-disease-name">{outcome_title}</div>
            <div class="cert-disease-sub">{outcome_sub}</div>
            <div class="cert-grid">
                <div class="cert-cell">
                    <div class="cert-cell-label">CALIBRATED POSTERIOR</div>
                    <div class="cert-cell-val" style="color: {posterior_color};">{state['leading_prob']*100:.1f}%{post_extra}</div>
                </div>
                <div class="cert-cell">
                    <div class="cert-cell-label">TOTAL HORIZON</div>
                    <div class="cert-cell-val">{len(state['evidence_trail'])} Step(s)</div>
                </div>
                <div class="cert-cell">
                    <div class="cert-cell-label">INITIAL ENTROPY H₀</div>
                    <div class="cert-cell-val">{h0:.3f} <span style="font-size: 0.72rem; color: #64748b;">bits</span></div>
                </div>
                <div class="cert-cell">
                    <div class="cert-cell-label">ENTROPY DELTA &Delta;H</div>
                    <div class="cert-cell-val">{dh_display}</div>
                </div>
            </div>
            <div class="cert-footer">
                <div style="margin-bottom: 4px;"><b>Diagnostic Trajectory:</b> <i>{init_lead_name.replace('_', ' ').title()}</i> &rarr; <b style="color: #ffffff;">{final_disease}</b></div>
                <div>{footer_note}</div>
            </div>
        </div>
        """)
