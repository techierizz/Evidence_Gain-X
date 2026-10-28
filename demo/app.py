import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import json
import torch
import pandas as pd

import importlib
import src.verification.reliability as rel_mod
import src.diagnosis.pipeline as pipe_mod
importlib.reload(rel_mod)
importlib.reload(pipe_mod)

from src.utils.config import load_config
from src.data.manifest import DatasetManifest, EvidenceItem
from src.models.classifier import DiseaseClassifier
from src.evidence.selector import EvidenceGainXPolicy, EvidenceGainSelector
from src.diagnosis.pipeline import EvidenceGainXPipeline

st.set_page_config(
    page_title="EvidenceGain-X | Next-Best Evidence Diagnostic Engine",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ULTRA-PREMIUM BIOLUMINESCENT DARK GLASSMORPHISM DESIGN SYSTEM
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Base Reset & Fonts */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Background & Ambient Glow */
    .stApp {
        background: radial-gradient(circle at 15% 15%, rgba(16, 185, 129, 0.08) 0%, transparent 40%),
                    radial-gradient(circle at 85% 85%, rgba(6, 182, 212, 0.06) 0%, transparent 40%),
                    #0c120f;
        color: #e2e8f0;
    }

    /* Hero Header */
    .hero-container {
        padding: 1.8rem 2.2rem;
        background: linear-gradient(135deg, rgba(18, 30, 24, 0.85) 0%, rgba(12, 22, 17, 0.95) 100%);
        border: 1px solid rgba(52, 211, 153, 0.2);
        border-radius: 18px;
        box-shadow: 0 12px 35px -10px rgba(0, 0, 0, 0.5), 0 0 20px rgba(16, 185, 129, 0.05);
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #10b981, #06b6d4, #10b981);
    }
    .hero-telemetry {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 4px 14px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #34d399;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.8; }
        50% { transform: scale(1.3); opacity: 1; box-shadow: 0 0 14px #34d399; }
        100% { transform: scale(0.95); opacity: 0.8; }
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #ffffff 40%, #a7f3d0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        max-width: 850px;
        line-height: 1.5;
        margin: 0;
    }

    /* Glassmorphic Cards */
    .glass-card {
        background: linear-gradient(145deg, rgba(22, 36, 29, 0.7) 0%, rgba(15, 25, 20, 0.85) 100%);
        border: 1px solid rgba(52, 211, 153, 0.18);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.4rem;
        backdrop-filter: blur(14px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(52, 211, 153, 0.35);
    }
    .card-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Metric Badges & Pills */
    .stat-pill {
        display: inline-block;
        padding: 3px 10px;
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-accept {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-downweight {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-reject {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* Custom Progress Bar for Probabilities */
    .prob-bar-container {
        margin: 8px 0;
    }
    .prob-bar-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .prob-bar-bg {
        height: 10px;
        background: rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease-out;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1612 !important;
        border-right: 1px solid rgba(52, 211, 153, 0.15);
    }
    section[data-testid="stSidebar"] p, 
    section[data-testid="stSidebar"] span, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }

    /* Metric Display Box */
    .kpi-box {
        text-align: center;
        padding: 0.8rem 0.5rem;
        background: rgba(10, 18, 14, 0.75);
        border: 1px solid rgba(52, 211, 153, 0.18);
        border-radius: 12px;
        min-height: 82px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .kpi-value {
        font-size: 1.25rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 3px;
        line-height: 1.2;
        word-break: normal;
    }
    .kpi-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Preset Buttons in Horizontal Block */
    div[data-testid="stHorizontalBlock"] button {
        background: rgba(18, 32, 25, 0.85) !important;
        border: 1px solid rgba(52, 211, 153, 0.25) !important;
        color: #f1f5f9 !important;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: rgba(16, 185, 129, 0.25) !important;
        border-color: #10b981 !important;
        color: #34d399 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
    }

    /* Trophy Card */
    .trophy-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.1) 100%);
        border: 1px solid rgba(52, 211, 153, 0.4);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.1);
    }

    /* Streamlit Widget Styling Overrides */
    div[data-testid="stMetricValue"] {
        color: #34d399 !important;
        font-weight: 800 !important;
    }
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }
    .stButton>button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.6);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# PIPELINE & MODEL INITIALIZATION
# ==============================================================================
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

# Load benchmark manifest
manifest_path = "data/manifests/dataset_manifest.json"
manifest = DatasetManifest.load_json(manifest_path) if Path(manifest_path).exists() else None

# ==============================================================================
# SIDEBAR CONTROLS & DIAGNOSTIC CONFIGURATION
# ==============================================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 1rem;">
            <span style="font-size: 1.6rem;">⚙️</span>
            <span style="font-size: 1.25rem; font-weight: 700; color: #f1f5f9;">Control Center</span>
        </div>
    """, unsafe_allow_html=True)
    
    input_mode = st.radio("Inference Input Source:", ["Interactive Live Upload", "Authentic Benchmark Gallery"])
    max_steps = st.slider("Evidence Budget Horizon ($B$):", 1, 5, cfg["evidence"]["max_budget"])
    pipeline.max_budget = max_steps
    
    st.markdown("---")
    if st.button("🔄 Reset Diagnostic Episode", use_container_width=True):
        st.session_state["current_state"] = None
        st.session_state["active_case"] = None
        st.session_state["evidence_history"] = []
        st.session_state["active_preset_name"] = None
        st.rerun()

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; margin-bottom: 8px;">
            Pathological Targets Monitored
        </div>
    """, unsafe_allow_html=True)
    
    disease_colors = {
        "early_blight": "#f59e0b",
        "late_blight": "#ef4444",
        "septoria_leaf_spot": "#8b5cf6",
        "healthy": "#10b981"
    }
    for c in classes:
        color = disease_colors.get(c, "#10b981")
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 4px 0;">
                <span style="font-size: 0.9rem; color: #e2e8f0;">{c.replace('_', ' ').title()}</span>
                <span style="width: 8px; height: 8px; border-radius: 50%; background: {color}; box-shadow: 0 0 6px {color};"></span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.78rem; color: #64748b; line-height: 1.4;">
            <b>Benchmark:</b> PlantVillage PV-Tomato<br>
            <b>Vision Backbone:</b> ResNet-18 (ImageNet Pre-trained)<br>
            <b>Calibration:</b> Temperature Scaling ($T=1.8928$)
        </div>
    """, unsafe_allow_html=True)

# Session State Initialization
if "current_state" not in st.session_state:
    st.session_state["current_state"] = None
if "active_case" not in st.session_state:
    st.session_state["active_case"] = None
if "active_preset_name" not in st.session_state:
    st.session_state["active_preset_name"] = None

# ==============================================================================
# HERO HEADER & TELEMETRY BANNER
# ==============================================================================
st.markdown(f"""
<div class="hero-container">
    <div class="hero-telemetry">
        <span class="pulse-dot"></span>
        <span>Active Vision Engine Online • PlantVillage Authentic Benchmark Active</span>
    </div>
    <div class="hero-title">🌿 EvidenceGain-X</div>
    <div class="hero-subtitle">
        Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis. 
        Transitions from passive image classification to goal-directed sequential inquiry when foliar diseases conflict.
    </div>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# QUICK 1-CLICK DEMONSTRATION PRESETS
# ==============================================================================
st.markdown("""
<div style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">
    ⚡ Instant Demonstration Presets (Click to Auto-Load Authentic Test Cases)
</div>
""", unsafe_allow_html=True)

p_col1, p_col2, p_col3, p_col4, p_col5, p_col6 = st.columns(6)

preset_to_load = None

with p_col1:
    if st.button("🍂 Early Blight", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_real_early_blight.jpg", "Authentic Early Blight (Alternaria solani)")
with p_col2:
    if st.button("💧 Late Blight", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_real_late_blight.jpg", "Authentic Late Blight (Phytophthora infestans)")
with p_col3:
    if st.button("🔬 Septoria Spot", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_real_septoria.jpg", "Authentic Septoria Leaf Spot (Look-alike)")
with p_col4:
    if st.button("🌿 Healthy Leaf", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_real_healthy.jpg", "Authentic Healthy Tomato Foliage")
with p_col5:
    if st.button("🌫️ Blurry Test", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_blurry.jpg", "Scenario C: Heavily Blurry Leaf (Defocus)")
with p_col6:
    if st.button("⚡ Contradiction", use_container_width=True):
        preset_to_load = ("data/test_samples/sample_healthy_contradiction.jpg", "Scenario B: Contradictory Input")

if preset_to_load:
    p_path, p_label = preset_to_load
    if Path(p_path).exists():
        loaded_img = Image.open(p_path).convert("RGB")
        st.session_state["current_state"] = pipeline.diagnose_initial(loaded_img)
        st.session_state["active_case"] = None
        st.session_state["active_preset_name"] = p_label
        st.rerun()

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==============================================================================
# MAIN TWO-COLUMN WORKFLOW LAYOUT
# ==============================================================================
col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header"><span>📸</span> 1. Macroscopic Foliar Observation ($X_0$)</div>', unsafe_allow_html=True)
    
    initial_img = None
    case_record = None

    if st.session_state.get("active_preset_name"):
        st.info(f"Loaded Preset: **{st.session_state['active_preset_name']}**")

    if input_mode == "Interactive Live Upload":
        uploaded_file = st.file_uploader("Upload leaf photograph (or capture via field camera):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            initial_img = Image.open(uploaded_file).convert("RGB")
            st.image(initial_img, caption="Live Uploaded Specimen", use_container_width=True)
            
            # Quality assessment of X0
            q_check = pipeline.reliability_checker.assess_image_quality(initial_img)
            if not q_check.is_acceptable:
                st.markdown(f'<div class="badge-reject" style="margin-top: 6px;">⚠️ Quality Warning: {", ".join(q_check.reasons)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-accept" style="margin-top: 6px;">✅ Sharp Focus (Laplacian: {q_check.blur_score:.1f})</div>', unsafe_allow_html=True)

            if st.button("🔍 Run Initial Diagnosis", type="primary", use_container_width=True):
                st.session_state["current_state"] = pipeline.diagnose_initial(initial_img)
                st.session_state["active_case"] = None
                st.session_state["active_preset_name"] = "Live User Upload"
                st.rerun()
    else:
        if manifest:
            case_ids = [c.case_id for c in manifest.cases]
            selected_id = st.selectbox("Select PlantVillage Benchmark Case:", case_ids)
            case_record = next(c for c in manifest.cases if c.case_id == selected_id)
            initial_img = Image.open(case_record.initial_image_path)
            st.image(initial_img, caption=f"Case ID: {case_record.case_id}", use_container_width=True)
            st.markdown(f"**Ground Truth Disease:** <span class='stat-pill'>{case_record.disease.replace('_', ' ').title()}</span>", unsafe_allow_html=True)
            
            if st.button("🔍 Run Diagnostic Inquiry", type="primary", use_container_width=True):
                st.session_state["current_state"] = pipeline.diagnose_initial(case_record.initial_image_path)
                st.session_state["active_case"] = case_record
                st.session_state["active_preset_name"] = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    state = st.session_state.get("current_state")
    if state is None:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem 1.5rem;">
            <span style="font-size: 3rem;">🌱</span>
            <h3 style="color: #cbd5e1; margin-top: 1rem;">Awaiting Initial Leaf Observation</h3>
            <p style="color: #64748b; font-size: 0.95rem; max-width: 450px; margin: 0 auto;">
                Select an authentic PlantVillage benchmark case, click a 1-click preset above, or upload any real tomato leaf image from the web to begin sequential inquiry.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ==============================================================================
        # PRELIMINARY DIAGNOSIS & CONFLICT ANALYSIS
        # ==============================================================================
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>🧠</span> 2. Provisional Diagnosis & Conflict Analysis</div>', unsafe_allow_html=True)
        
        # KPI Row
        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value" style="color: #34d399;">{state['leading_disease'].replace('_', ' ').title()}</div>
                <div class="kpi-label">Provisional Hypothesis ($h_1$)</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{state['leading_prob']*100:.1f}%</div>
                <div class="kpi-label">Calibrated Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{state['entropy']:.3f}</div>
                <div class="kpi-label">Shannon Entropy (bits)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Belief Distribution Bars
        st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;'>Current Posterior Belief Distribution</div>", unsafe_allow_html=True)
        
        for i, c_name in enumerate(classes):
            p_val = state["current_probs"][i]
            bar_color = disease_colors.get(c_name, "#10b981")
            is_lead = (c_name == state["leading_disease"])
            border_glow = f"box-shadow: 0 0 10px {bar_color}40;" if is_lead else ""
            
            st.markdown(f"""
            <div class="prob-bar-container">
                <div class="prob-bar-header">
                    <span style="color: {'#ffffff' if is_lead else '#cbd5e1'}; font-weight: {'700' if is_lead else '500'};">
                        {c_name.replace('_', ' ').title()} {'⭐' if is_lead else ''}
                    </span>
                    <span style="color: {bar_color}; font-family: 'JetBrains Mono'; font-weight: 700;">
                        {p_val*100:.1f}%
                    </span>
                </div>
                <div class="prob-bar-bg" style="{border_glow}">
                    <div class="prob-bar-fill" style="width: {p_val*100:.1f}%; background: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Look-alike Conflict Analysis Box
        rival = state.get("rival_disease")
        margin = state.get("margin", 1.0)
        
        if state["is_ambiguous"] and rival:
            st.markdown(f"""
            <div style="margin-top: 14px; padding: 12px 16px; background: rgba(245, 158, 11, 0.12); border-left: 4px solid #f59e0b; border-radius: 8px;">
                <div style="font-weight: 700; color: #fbbf24; font-size: 0.95rem; margin-bottom: 4px;">
                    ⚠️ Look-Alike Ambiguity Detected: <i>{state['leading_disease'].replace('_', ' ').title()}</i> vs <i>{rival.replace('_', ' ').title()}</i>
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
                    Top hypothesis margin is narrow (Δp = {margin:.2f}). Standard passive CNNs would guess prematurely. EvidenceGain-X initiates active inquiry to isolate discriminative traits.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="margin-top: 14px; padding: 10px 14px; background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; border-radius: 8px; font-size: 0.88rem; color: #a7f3d0;">
                ✅ Decisive Hypothesis Separation: <b>{state['leading_disease'].replace('_', ' ').title()}</b> holds dominant margin (Δp = {margin:.2f}).
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ==============================================================================
        # NEXT-BEST EVIDENCE RECOMMENDATION (MODULES 4, 5, 6)
        # ==============================================================================
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header"><span>🎯</span> 3. Next-Best Evidence Recommendation ($e^*$)</div>', unsafe_allow_html=True)

        used_cands = [item["evidence_type"] for item in state["evidence_trail"]]
        rem_cands = [c for c in cand_types if c not in used_cands]

        if not rem_cands:
            st.info("All 5 candidate evidence modalities have been exhausted for this episode.")
        elif state["entropy"] <= pipeline.stopping_entropy and state["margin"] >= pipeline.stopping_margin:
            st.markdown("""
            <div style="padding: 14px; background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; border-radius: 10px; text-align: center;">
                <span style="font-size: 1.5rem;">🎯</span>
                <div style="font-weight: 700; color: #34d399; font-size: 1rem; margin-top: 4px;">
                    Stopping Criterion Reached ($H < 0.35$ bits)
                </div>
                <div style="font-size: 0.85rem; color: #cbd5e1;">Diagnostic uncertainty is sufficiently resolved. No further evidence required.</div>
            </div>
            """, unsafe_allow_html=True)
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

            st.markdown(f"""
            <div style="padding: 12px 18px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.12) 100%); border: 1px solid #10b981; border-radius: 12px; margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 0.75rem; text-transform: uppercase; font-weight: 700; color: #34d399; letter-spacing: 0.05em;">Optimal Target Observation (e*)</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #ffffff;">{best_cand.replace('_', ' ').title()}</div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94a3b8;">Max Net Utility U(e)</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono';">{best_cand_row['total_utility']:.3f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Styled Utility Table
            st.dataframe(
                u_df[["candidate", "hypothesis_separation", "uncertainty_reduction", "cost", "total_utility"]].rename(
                    columns={
                        "candidate": "Modality",
                        "hypothesis_separation": "Separation (JS)",
                        "uncertainty_reduction": "Exp. ΔH",
                        "cost": "Cost",
                        "total_utility": "Net Utility U(e)"
                    }
                ).style.highlight_max(subset=["Net Utility U(e)"], color="#065f46"),
                use_container_width=True
            )

            # Interactive Acquisition Expander
            with st.expander(f"📥 Acquire Requested Evidence: {best_cand.replace('_', ' ').title()}", expanded=True):
                ev_to_submit = None
                
                active_case = st.session_state.get("active_case")
                if active_case and best_cand in active_case.available_evidence:
                    b_item = active_case.available_evidence[best_cand]
                    st.markdown(f"**Benchmark Observation Ready:** `{b_item.evidence_type}` ({b_item.modality})")
                    if st.button(f"⚡ Use Associated Benchmark Evidence ({best_cand.replace('_', ' ').title()})", type="primary"):
                        ev_to_submit = b_item

                st.markdown("<div style='margin-top: 10px; font-weight: 600; color: #94a3b8;'>Or upload live evidence observation:</div>", unsafe_allow_html=True)
                
                if best_cand in ["lesion_closeup", "leaf_underside", "stem_view"]:
                    live_ev_file = st.file_uploader(f"Upload {best_cand.replace('_', ' ').title()} Image:", type=["jpg", "png", "jpeg"], key=f"live_{best_cand}")
                    if live_ev_file:
                        live_pil = Image.open(live_ev_file).convert("RGB")
                        st.image(live_pil, width=220)
                        if st.button("Submit Live Image Evidence", type="primary"):
                            ev_to_submit = EvidenceItem(
                                evidence_type=best_cand,
                                modality="image",
                                data_path_or_content=live_pil,
                                cost=1.0
                            )
                else:
                    text_input = st.text_area("Field Symptom Description / Microclimate Context:", value="Dark brown concentric rings with chlorotic yellow halo.")
                    if st.button("Submit Textual Observation", type="primary"):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="text",
                            data_path_or_content=text_input,
                            cost=0.5
                        )

                if ev_to_submit is not None:
                    with st.spinner("Executing Reliability Gating, Contradiction Verification & Bayesian Update..."):
                        updated_state = pipeline.process_single_evidence(state, best_cand, ev_to_submit)
                        st.session_state["current_state"] = updated_state
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# 4. SELF-VERIFYING AUDIT TRAIL & FINAL VERIFIED DIAGNOSIS
# ==============================================================================
if state and len(state["evidence_trail"]) > 0:
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown("### 🛡️ 4. Self-Verifying Audit Trail & Diagnostic Trajectory")
    
    col_t1, col_t2 = st.columns([1.5, 1], gap="large")
    
    with col_t1:
        for item in state["evidence_trail"]:
            act = item.get("action", "ACCEPT")
            badge = f'<span class="badge-accept">ACCEPT</span>' if act == "ACCEPT" else (f'<span class="badge-downweight">DOWNWEIGHT</span>' if act == "DOWNWEIGHT" else f'<span class="badge-reject">REJECT</span>')
            
            with st.expander(f"Step {item['step']}: Acquired '{item['evidence_type'].replace('_', ' ').title()}' — Action: {act}", expanded=True):
                st.markdown(f"**Reliability Gating:** {badge} &nbsp;|&nbsp; Quality Score: `{item['quality_score']:.2f}`", unsafe_allow_html=True)
                if item.get("reasons"):
                    st.markdown(f"- **Quality Diagnostics:** {', '.join(item['reasons'])}")
                
                is_rejected = act in ["REJECT", "REJECTED"]
                if is_rejected:
                    st.error("❌ **Observation Blocked & Discarded:** Image quality is unacceptable. The observation was rejected by Module 7 (Reliability Gating) and blocked from corrupting the diagnostic state. Previous beliefs were safely preserved.")
                else:
                    is_contra = item.get("contradiction", False)
                    if is_contra:
                        st.warning(f"⚠️ **Contradiction Detected:** {item.get('contradiction_explanation')}")
                    else:
                        st.success("✅ **Observation Verified:** Observation conforms with counterfactual expectation trajectory.")
                
                st.markdown(f"**Updated Belief:** <span class='stat-pill'>{item['leading_after'].replace('_', ' ').title()}</span> ({item['leading_prob_after']*100:.1f}%) &nbsp;|&nbsp; **Entropy:** `{item['entropy_after']:.3f} bits`", unsafe_allow_html=True)

    with col_t2:
        st.markdown('<div class="trophy-card">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
            <span style="font-size: 1.8rem;">🏆</span>
            <span style="font-size: 1.2rem; font-weight: 800; color: #ffffff;">Final Verified Diagnosis</span>
        </div>
        """, unsafe_allow_html=True)
        
        final_disease = state['leading_disease'].replace('_', ' ').title()
        st.markdown(f"""
        <div style="font-size: 1.6rem; font-weight: 800; color: #34d399; margin-bottom: 8px;">
            {final_disease}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"**Final Confidence:** `{state['leading_prob']*100:.1f}%`")
        st.markdown(f"**Total Acquisition Steps:** `{len(state['evidence_trail'])}`")
        
        init_lead_idx = int(np.argmax(state["initial_probs"]))
        init_lead_name = classes[init_lead_idx]
        
        # Real entropy reduction ΔH = H0 - Ht
        init_p = np.array(state["initial_probs"])
        h0 = float(-np.sum(init_p * np.log2(init_p + 1e-12)))
        ht = float(state["entropy"])
        delta_h = max(0.0, h0 - ht)
        
        st.markdown(f"**Diagnostic Shift:** *{init_lead_name.replace('_', ' ').title()}* ➔ **{final_disease}**")
        st.markdown(f"**Initial Entropy ($H_0$):** `{h0:.3f} bits`")
        st.markdown(f"**Current Entropy ($H_t$):** `{ht:.3f} bits`")
        st.markdown(f"**Total Uncertainty Reduction ($\\\\Delta H$):** <span class='stat-pill' style='background: rgba(6, 182, 212, 0.2); color: #38bdf8;'>{delta_h:.3f} bits</span>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
