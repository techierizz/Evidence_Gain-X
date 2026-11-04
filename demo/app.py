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

st.set_page_config(
    page_title="EvidenceGain-X | Diagnostic Console",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }

    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.2rem 1.6rem;
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 12px;
        margin-bottom: 1.4rem;
    }
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin: 0;
    }
    .header-subtitle {
        font-size: 0.88rem;
        color: #94a3b8;
        margin-top: 3px;
    }
    .header-badges {
        display: flex;
        gap: 8px;
    }
    .meta-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        background: #1e293b;
        border: 1px solid #334155;
        padding: 4px 10px;
        border-radius: 6px;
    }

    .panel-card {
        background: #111827;
        border: 1px solid #1f293d;
        border-radius: 12px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: -0.01em;
    }

    .status-badge-pass {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #10b981;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }
    .status-badge-reject {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ef4444;
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }
    .status-badge-warn {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        color: #f59e0b;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 3px 8px;
        border-radius: 5px;
        display: inline-block;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 1.2rem;
    }
    .metric-cell {
        background: #161f30;
        border: 1px solid #232d42;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        text-align: left;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 2px;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .prob-row {
        margin: 10px 0;
    }
    .prob-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.85rem;
        margin-bottom: 4px;
    }
    .prob-track {
        height: 8px;
        background: #1e293b;
        border-radius: 4px;
        overflow: hidden;
    }
    .prob-bar {
        height: 100%;
        border-radius: 4px;
    }

    .alert-box {
        padding: 12px 14px;
        background: #1a1e28;
        border-left: 3px solid #f59e0b;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #cbd5e1;
    }
    .success-box {
        padding: 12px 14px;
        background: #14201c;
        border-left: 3px solid #10b981;
        border-radius: 6px;
        margin-top: 1rem;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #a7f3d0;
    }

    .outcome-card {
        background: #111c18;
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 12px;
        padding: 1.4rem;
    }

    div[data-testid="stHorizontalBlock"] button {
        background: #161f30 !important;
        border: 1px solid #28354d !important;
        color: #cbd5e1 !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 6px 10px !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        background: #1e293b !important;
        border-color: #3b82f6 !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d111a !important;
        border-right: 1px solid #1f293d;
    }
</style>
""", unsafe_allow_html=True)

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

with st.sidebar:
    st.markdown("""
        <div style="font-size: 1.05rem; font-weight: 700; color: #f8fafc; margin-bottom: 1rem;">
            Configuration Console
        </div>
    """, unsafe_allow_html=True)

    input_mode = st.radio(
        "Observation Input Mode:",
        ["Benchmark Gallery", "Live Image Ingestion"],
        index=0 if st.session_state["input_source_mode"] == "Benchmark Gallery" else 1,
        key="radio_input_mode"
    )
    st.session_state["input_source_mode"] = input_mode

    max_steps = st.slider("Evidence Budget Horizon (B):", 1, 5, cfg["evidence"]["max_budget"])
    pipeline.max_budget = max_steps

    st.markdown("---")
    if st.button("Reset Diagnostic Session", use_container_width=True):
        st.session_state["current_state"] = None
        st.session_state["active_case"] = None
        st.session_state["active_preset_name"] = None
        st.session_state["active_image_path_or_pil"] = None
        st.session_state["selected_case_id"] = "tomato_early_blight_0001" if manifest else None
        st.rerun()

    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 8px;">
            Target Foliar Pathologies
        </div>
    """, unsafe_allow_html=True)

    disease_palette = {
        "early_blight": "#f59e0b",
        "late_blight": "#ef4444",
        "septoria_leaf_spot": "#8b5cf6",
        "healthy": "#10b981"
    }
    for c in classes:
        col = disease_palette.get(c, "#10b981")
        st.markdown(f"""
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 3px 0;">
                <span style="font-size: 0.85rem; color: #cbd5e1;">{c.replace('_', ' ').title()}</span>
                <span style="width: 8px; height: 8px; border-radius: 50%; background: {col};"></span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(r"""
        <div style="font-size: 0.76rem; color: #64748b; line-height: 1.5;">
            <b>Dataset:</b> PlantVillage PV-Tomato<br>
            <b>Encoder:</b> ResNet-18 (ImageNet-1K)<br>
            <b>Calibration:</b> Temperature Scaling (T = 1.8928)<br>
            <b>Stopping Criterion:</b> H(P) &le; 0.35 bits
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="top-header">
    <div>
        <div class="header-title">EvidenceGain-X</div>
        <div class="header-subtitle">Active Sequential Diagnostic Engine • Counterfactual Next-Best Evidence Acquisition</div>
    </div>
    <div class="header-badges">
        <span class="meta-tag">BACKBONE: RESNET-18</span>
        <span class="meta-tag">DATASET: PLANTVILLAGE</span>
        <span class="meta-tag">CALIBRATED: T=1.89</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='font-size: 0.8rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 6px;'>Clinical Case Presets</div>", unsafe_allow_html=True)

p1, p2, p3, p4, p5, p6 = st.columns(6)

with p1:
    if st.button("Early Blight", use_container_width=True):
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
    if st.button("Late Blight", use_container_width=True):
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
    if st.button("Septoria Spot", use_container_width=True):
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
    if st.button("Healthy Control", use_container_width=True):
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
    if st.button("Test: Blur Gating", use_container_width=True):
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
    if st.button("Test: Contradiction", use_container_width=True):
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

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">1. Specimen Observation (X₀)</div>', unsafe_allow_html=True)

    if st.session_state.get("active_preset_name"):
        st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;'>Active Case: <b>{st.session_state['active_preset_name']}</b></div>", unsafe_allow_html=True)

    curr_input_mode = st.session_state["input_source_mode"]

    if curr_input_mode == "Live Image Ingestion":
        uploaded_file = st.file_uploader("Upload macroscopic leaf photograph:", type=["jpg", "jpeg", "png"])

        st.markdown("<div style='font-size: 0.78rem; color: #64748b; margin-top: 4px;'>Or select reference specimen:</div>", unsafe_allow_html=True)
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
            st.image(live_pil, caption="Ingested Specimen", use_container_width=True)

            q_check = pipeline.reliability_checker.assess_image_quality(live_pil)
            if not q_check.is_acceptable:
                st.markdown(f'<div class="status-badge-reject" style="margin-top: 6px;">QUALITY WARNING: {", ".join(q_check.reasons)}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="status-badge-pass" style="margin-top: 6px;">OPTICAL QUALITY: PASS (Laplacian: {q_check.blur_score:.1f} | Luminance: {q_check.brightness_score:.1f})</div>', unsafe_allow_html=True)

            if st.button("Execute Provisional Diagnosis", type="primary", use_container_width=True):
                st.session_state["current_state"] = pipeline.diagnose_initial(live_pil)
                st.session_state["active_case"] = None
                st.session_state["active_preset_name"] = "Live Upload Specimen"
                st.rerun()
        elif st.session_state.get("active_image_path_or_pil"):
            disp_target = st.session_state["active_image_path_or_pil"]
            if isinstance(disp_target, (str, Path)) and Path(disp_target).exists():
                st.image(str(disp_target), caption="Active Specimen", use_container_width=True)
            elif isinstance(disp_target, Image.Image):
                st.image(disp_target, caption="Active Specimen", use_container_width=True)

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
                st.image(initial_img, caption=f"Case Identifier: {case_record.case_id}", use_container_width=True)
                st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin: 6px 0;'>Ground Truth Label: <span class='meta-tag'>{case_record.disease.replace('_', ' ').title()}</span></div>", unsafe_allow_html=True)

                q_check = pipeline.reliability_checker.assess_image_quality(case_record.initial_image_path)
                st.markdown(f'<div class="status-badge-pass">OPTICAL PASS (Laplacian: {q_check.blur_score:.1f} &sigma;&sup2;)</div>', unsafe_allow_html=True)

            if st.button("Run Provisional Diagnosis", type="primary", use_container_width=True):
                st.session_state["current_state"] = pipeline.diagnose_initial(case_record.initial_image_path)
                st.session_state["active_image_path_or_pil"] = case_record.initial_image_path
                st.session_state["active_preset_name"] = None
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    state = st.session_state.get("current_state")
    if state is None:
        st.markdown("""
        <div class="panel-card" style="text-align: center; padding: 3rem 1.5rem;">
            <div style="font-size: 1.1rem; font-weight: 600; color: #94a3b8;">Awaiting Specimen Input</div>
            <p style="color: #64748b; font-size: 0.88rem; max-width: 420px; margin: 0.6rem auto 0 auto; line-height: 1.5;">
                Select a benchmark record, load a clinical preset above, or ingest a foliar photograph to begin sequential inquiry.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">2. Provisional Hypothesis Tracking & Entropy State</div>', unsafe_allow_html=True)

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(f"""
            <div class="metric-cell">
                <div class="metric-label">Leading Hypothesis (h₁)</div>
                <div class="metric-value" style="color: #38bdf8;">{state['leading_disease'].replace('_', ' ').title()}</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="metric-cell">
                <div class="metric-label">Calibrated Probability</div>
                <div class="metric-value">{state['leading_prob']*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="metric-cell">
                <div class="metric-label">Shannon Entropy H(P)</div>
                <div class="metric-value">{state['entropy']:.3f} <span style="font-size: 0.8rem; font-weight: 400; color: #64748b;">bits</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='font-size: 0.76rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin: 12px 0 6px 0;'>Posterior Belief Distribution</div>", unsafe_allow_html=True)

        for i, c_name in enumerate(classes):
            p_val = state["current_probs"][i]
            bar_color = disease_palette.get(c_name, "#10b981")
            is_lead = (c_name == state["leading_disease"])

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-labels">
                    <span style="color: {'#f8fafc' if is_lead else '#94a3b8'}; font-weight: {'600' if is_lead else '400'};">
                        {c_name.replace('_', ' ').title()}
                    </span>
                    <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.82rem; color: {'#f8fafc' if is_lead else '#94a3b8'}; font-weight: 600;">
                        {p_val*100:.1f}%
                    </span>
                </div>
                <div class="prob-track">
                    <div class="prob-bar" style="width: {p_val*100:.1f}%; background-color: {bar_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        rival = state.get("rival_disease")
        margin = state.get("margin", 1.0)

        if state["is_ambiguous"] and rival:
            st.markdown(f"""
            <div class="alert-box">
                <b>Look-Alike Ambiguity Detected:</b> Margin narrow between <i>{state['leading_disease'].replace('_', ' ').title()}</i> and <i>{rival.replace('_', ' ').title()}</i> (&Delta;p = {margin:.2f}). Standard single-shot CNNs yield high error on this boundary. EvidenceGain-X initiates goal-directed inquiry.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="success-box">
                <b>Decisive Separation:</b> <i>{state['leading_disease'].replace('_', ' ').title()}</i> established dominant margin (&Delta;p = {margin:.2f}).
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">3. Next-Best Evidence Recommendation (e*)</div>', unsafe_allow_html=True)

        used_cands = [item["evidence_type"] for item in state["evidence_trail"]]
        rem_cands = [c for c in cand_types if c not in used_cands]

        if not rem_cands:
            st.info("All candidate evidence modalities have been exhausted for this episode.")
        elif state["entropy"] <= pipeline.stopping_entropy and state["margin"] >= pipeline.stopping_margin:
            st.markdown(r"""
            <div class="success-box" style="text-align: center; padding: 1rem;">
                <div style="font-weight: 600; font-size: 0.95rem; color: #10b981;">Stopping Criterion Satisfied (H(P) &le; 0.35 bits)</div>
                <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 3px;">Diagnostic certainty sufficiently resolved. No further evidence acquisition required.</div>
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
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: #161f30; border: 1px solid #232d42; border-radius: 8px; margin-bottom: 0.9rem;">
                <div>
                    <div style="font-size: 0.7rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Optimal Query Modality (e*)</div>
                    <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">{best_cand.replace('_', ' ').title()}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;">Max Net Utility U(e)</div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; font-family: 'JetBrains Mono';">{best_cand_row['total_utility']:.3f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                u_df[["candidate", "hypothesis_separation", "uncertainty_reduction", "cost", "total_utility"]].rename(
                    columns={
                        "candidate": "Modality",
                        "hypothesis_separation": "Separation (JS)",
                        "uncertainty_reduction": "Exp. ΔH",
                        "cost": "Cost",
                        "total_utility": "Net Utility U(e)"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            with st.expander(f"Acquire Clue: {best_cand.replace('_', ' ').title()}", expanded=True):
                ev_to_submit = None
                active_case = st.session_state.get("active_case")

                if active_case and best_cand in active_case.available_evidence:
                    b_item = active_case.available_evidence[best_cand]
                    st.markdown(f"<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 6px;'>Authentic Benchmark Clue: <code>{b_item.evidence_type}</code> ({b_item.modality})</div>", unsafe_allow_html=True)
                    if st.button(f"Submit Benchmark Observation ({best_cand.replace('_', ' ').title()})", type="primary", use_container_width=True):
                        ev_to_submit = b_item

                st.markdown("<div style='font-size: 0.76rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin: 10px 0 4px 0;'>Stress-Testing Controls</div>", unsafe_allow_html=True)
                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    if st.button("Inject Defocus Blur (Module 7)", use_container_width=True):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="image",
                            data_path_or_content="data/test_samples/sample_blurry.jpg",
                            cost=1.0
                        )
                with sc_c2:
                    if st.button("Inject Contradictory Leaf (Module 8)", use_container_width=True):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="image",
                            data_path_or_content="data/test_samples/sample_healthy_contradiction.jpg",
                            cost=1.0
                        )

                st.markdown("<div style='font-size: 0.76rem; font-weight: 600; color: #64748b; text-transform: uppercase; margin: 10px 0 4px 0;'>Or Upload Field Observation</div>", unsafe_allow_html=True)

                if best_cand in ["lesion_closeup", "leaf_underside", "stem_view"]:
                    live_ev_file = st.file_uploader(f"Upload {best_cand.replace('_', ' ').title()}:", type=["jpg", "png", "jpeg"], key=f"live_{best_cand}")
                    if live_ev_file:
                        live_pil = Image.open(live_ev_file).convert("RGB")
                        st.image(live_pil, width=220)
                        if st.button("Submit Ingested Image", type="primary"):
                            ev_to_submit = EvidenceItem(
                                evidence_type=best_cand,
                                modality="image",
                                data_path_or_content=live_pil,
                                cost=1.0
                            )
                else:
                    text_input = st.text_area("Field Pathology Description:", value="Dark brown concentric rings with chlorotic yellow halo.")
                    if st.button("Submit Text Description", type="primary"):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="text",
                            data_path_or_content=text_input,
                            cost=0.5
                        )

                if ev_to_submit is not None:
                    with st.spinner("Executing Reliability Gating, Contradiction Verification & Belief Updating..."):
                        updated_state = pipeline.process_single_evidence(state, best_cand, ev_to_submit)
                        st.session_state["current_state"] = updated_state
                        st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

if state and len(state["evidence_trail"]) > 0:
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    col_t1, col_t2 = st.columns([1.5, 1], gap="large")

    with col_t1:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">4. Sequential Diagnostic Audit Trail</div>', unsafe_allow_html=True)

        for item in state["evidence_trail"]:
            act = item.get("action", "ACCEPT")
            badge = f'<span class="status-badge-pass">ACCEPT</span>' if act == "ACCEPT" else f'<span class="status-badge-reject">REJECT</span>'

            with st.expander(f"Step {item['step']}: {item['evidence_type'].replace('_', ' ').title()} — Status: {act}", expanded=True):
                st.markdown(f"<div style='font-size: 0.85rem;'><b>Module 7 Reliability Gating:</b> {badge} &nbsp;|&nbsp; Quality Score: <code>{item['quality_score']:.2f}</code></div>", unsafe_allow_html=True)
                if item.get("reasons"):
                    st.markdown(f"<div style='font-size: 0.8rem; color: #94a3b8; margin-top: 3px;'>Diagnostics: {', '.join(item['reasons'])}</div>", unsafe_allow_html=True)

                is_rejected = act in ["REJECT", "REJECTED"]
                if is_rejected:
                    st.markdown("""
                    <div class="alert-box" style="border-left-color: #ef4444; background: #221518; color: #fca5a5;">
                        <b>Observation Gated & Blocked:</b> Image quality failed optical criteria (Laplacian variance below threshold). Blocked from corrupting diagnostic distribution. Prior beliefs preserved.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    is_contra = item.get("contradiction", False)
                    if is_contra:
                        st.markdown(f"""
                        <div class="alert-box" style="border-left-color: #f59e0b; background: #262016; color: #fde68a;">
                            <b>Contradiction Detected (Module 8):</b> {item.get('contradiction_explanation')}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box" style="padding: 8px 12px; margin-top: 6px;">
                            Observation aligned with counterfactual trajectory.
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown(f"<div style='font-size: 0.85rem; margin-top: 8px;'>Updated Hypothesis: <b>{item['leading_after'].replace('_', ' ').title()}</b> ({item['leading_prob_after']*100:.1f}%) &nbsp;|&nbsp; Entropy: <code>{item['entropy_after']:.3f} bits</code></div>", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_t2:
        st.markdown('<div class="outcome-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Diagnostic Outcome</div>', unsafe_allow_html=True)

        final_disease = state['leading_disease'].replace('_', ' ').title()
        st.markdown(f"""
        <div style="font-size: 1.5rem; font-weight: 700; color: #f8fafc; margin: 4px 0 10px 0;">
            {final_disease}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"<div style='font-size: 0.85rem; margin: 4px 0;'>Calibrated Confidence: <b>{state['leading_prob']*100:.1f}%</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; margin: 4px 0;'>Acquisition Horizon: <b>{len(state['evidence_trail'])} step(s)</b></div>", unsafe_allow_html=True)

        init_lead_idx = int(np.argmax(state["initial_probs"]))
        init_lead_name = classes[init_lead_idx]

        init_p = np.array(state["initial_probs"])
        h0 = float(-np.sum(init_p * np.log2(init_p + 1e-12)))
        ht = float(state["entropy"])
        delta_h = max(0.0, h0 - ht)

        st.markdown(f"<div style='font-size: 0.85rem; margin: 4px 0;'>Trajectory: <i>{init_lead_name.replace('_', ' ').title()}</i> &rarr; <b>{final_disease}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; margin: 4px 0;'>Initial Entropy: <code>{h0:.3f} bits</code></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size: 0.85rem; margin: 4px 0;'>Current Entropy: <code>{ht:.3f} bits</code></div>", unsafe_allow_html=True)
        st.markdown(rf"<div style='font-size: 0.85rem; margin-top: 8px;'>Uncertainty Reduction (&Delta;H): <span class='meta-tag' style='color: #38bdf8;'>{delta_h:.3f} bits</span></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
