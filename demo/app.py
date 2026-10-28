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
    page_title="EvidenceGain-X | Crop Diagnostic Decision Support",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3d2f; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.05rem; color: #4a6b5a; margin-bottom: 1.5rem; }
    .metric-card { background: #f8faf9; border-radius: 8px; padding: 12px 18px; border-left: 4px solid #2e7d32; }
    .badge-accept { background: #e8f5e9; color: #2e7d32; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-downweight { background: #fff8e1; color: #f57f17; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
    .badge-reject { background: #ffebee; color: #c62828; padding: 3px 8px; border-radius: 4px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌿 EvidenceGain-X</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis</div>',
    unsafe_allow_html=True
)

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

# Load benchmark manifest if available
manifest_path = "data/manifests/dataset_manifest.json"
manifest = DatasetManifest.load_json(manifest_path) if Path(manifest_path).exists() else None

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Pipeline Configuration")
    input_mode = st.radio("Input Source Mode:", ["Interactive Live Upload", "Benchmark Case Gallery"])
    max_steps = st.slider("Max Evidence Budget Steps:", 1, 5, cfg["evidence"]["max_budget"])
    pipeline.max_budget = max_steps
    
    if st.button("🔄 Reset Diagnostic Episode", use_container_width=True):
        st.session_state["current_state"] = None
        st.session_state["active_case"] = None
        st.session_state["evidence_history"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Target Crop:** `Tomato (Solanum lycopersicum)`")
    st.markdown("**Diseases Monitored:**")
    for c in classes:
        st.markdown(f"- *{c.replace('_', ' ').title()}*")
    st.markdown("---")
    st.caption("EvidenceGain-X Core Principles: Sequential Active Vision • Multi-factor Evidence Utility • Self-Verifying Contradiction Gating")

# Session State Initialization
if "current_state" not in st.session_state:
    st.session_state["current_state"] = None
if "active_case" not in st.session_state:
    st.session_state["active_case"] = None
if "evidence_history" not in st.session_state:
    st.session_state["evidence_history"] = []

# Main Layout
col_obs, col_diag = st.columns([1, 1.2], gap="large")

with col_obs:
    st.subheader("1. Initial Crop Observation ($X_0$)")
    initial_img = None
    case_record = None

    if input_mode == "Interactive Live Upload":
        uploaded_file = st.file_uploader("Upload leaf photograph (JPG/PNG):", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            initial_img = Image.open(uploaded_file).convert("RGB")
            st.image(initial_img, caption="Live User Upload", use_container_width=True)
            if st.button("🔍 Run Initial Diagnosis", type="primary"):
                st.session_state["current_state"] = pipeline.diagnose_initial(initial_img)
                st.session_state["active_case"] = None
                st.session_state["evidence_history"] = []
    else:
        if manifest:
            case_ids = [c.case_id for c in manifest.cases]
            selected_id = st.selectbox("Select Benchmark Case:", case_ids)
            case_record = next(c for c in manifest.cases if c.case_id == selected_id)
            initial_img = Image.open(case_record.initial_image_path)
            st.image(initial_img, caption=f"Benchmark Case: {case_record.case_id}", use_container_width=True)
            st.info(f"**True Ground Truth Disease:** `{case_record.disease.replace('_', ' ').title()}`")
            
            if st.button("🔍 Run Diagnostic Inquiry", type="primary"):
                st.session_state["current_state"] = pipeline.diagnose_initial(case_record.initial_image_path)
                st.session_state["active_case"] = case_record
                st.session_state["evidence_history"] = []
        else:
            st.warning("No benchmark manifest found. Please run `python scripts/download_data.py` first.")

with col_diag:
    state = st.session_state.get("current_state")
    if state is not None:
        st.subheader("2. Preliminary Diagnosis & Conflict Analysis")
        
        # Key Metrics Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Provisional Diagnosis", state["leading_disease"].replace("_", " ").title())
        with m2:
            st.metric("Confidence", f"{state['leading_prob']*100:.1f}%")
        with m3:
            st.metric("Shannon Entropy", f"{state['entropy']:.3f} bits")

        # Probability Distribution Bar Chart
        probs_df = pd.DataFrame({
            "Disease": [c.replace("_", " ").title() for c in classes],
            "Probability": [state["current_probs"][i] for i in range(len(classes))]
        }).set_index("Disease")
        st.bar_chart(probs_df, y="Probability", height=200)

        # Competing Hypotheses & Conflict Callout
        rival = state.get("rival_disease")
        margin = state.get("margin", 1.0)
        
        if state["is_ambiguous"] and rival:
            st.warning(
                f"⚠️ **Ambiguity Detected:** Primary hypothesis **{state['leading_disease'].replace('_', ' ').title()}** "
                f"({state['leading_prob']*100:.1f}%) is in close conflict with critical rival **{rival.replace('_', ' ').title()}** "
                f"({state['rival_prob']*100:.1f}%) | Probability Margin: `{margin:.2f}`\n\n"
                f"*The system transitions from 'What disease is this?' to 'What evidence best separates these two candidates?'*"
            )
        else:
            st.success(f"✅ **Clear Distinction:** Leading hypothesis has strong margin (`{margin:.2f}`).")

        st.markdown("---")
        
        # 3. Next-Best Evidence Planning (Modules 4, 5, 6)
        st.subheader("3. Next-Best Evidence Recommendation ($e^*$)")
        
        # Calculate utilities for remaining candidates
        used_cands = [item["evidence_type"] for item in state["evidence_trail"]]
        rem_cands = [c for c in cand_types if c not in used_cands]

        if not rem_cands:
            st.info("All candidate evidence types have been exhausted.")
        elif state["entropy"] <= pipeline.stopping_entropy and state["margin"] >= pipeline.stopping_margin:
            st.success("🎯 **Stopping Criterion Reached:** Uncertainty is sufficiently resolved. No further evidence required.")
        else:
            # Table of utility scores
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

            st.dataframe(
                u_df[["candidate", "hypothesis_separation", "uncertainty_reduction", "cost", "risk", "total_utility"]].rename(
                    columns={
                        "candidate": "Candidate Evidence",
                        "hypothesis_separation": "Separation",
                        "uncertainty_reduction": "Exp. Δ Entropy",
                        "cost": "Cost",
                        "risk": "Risk",
                        "total_utility": "Net Utility U(e)"
                    }
                ).style.highlight_max(subset=["Net Utility U(e)"], color="#c8e6c9"),
                use_container_width=True
            )

            st.markdown(
                f"👉 **Optimal Observation:** **`{best_cand.replace('_', ' ').title()}`** "
                f"(Net Utility: `{best_cand_row['total_utility']:.3f}`) — expected to maximize separation between "
                f"*{state['leading_disease'].replace('_', ' ').title()}* and *{h2_name.replace('_', ' ').title()}*."
            )

            # Interactive Follow-up Evidence Input
            with st.expander(f"📥 Provide Requested Evidence: {best_cand.replace('_', ' ').title()}", expanded=True):
                ev_to_submit = None
                
                # Check if benchmark case has it
                active_case = st.session_state.get("active_case")
                if active_case and best_cand in active_case.available_evidence:
                    b_item = active_case.available_evidence[best_cand]
                    st.write(f"Benchmark contains observation: `{b_item.evidence_type}` ({b_item.modality})")
                    if st.button(f"Use Associated Observation ({best_cand})"):
                        ev_to_submit = b_item

                st.write("**Or provide live observation:**")
                if best_cand in ["lesion_closeup", "leaf_underside", "stem_view"]:
                    live_ev_file = st.file_uploader(f"Upload {best_cand.replace('_', ' ').title()} Image:", type=["jpg", "png"], key=f"live_{best_cand}")
                    if live_ev_file:
                        live_pil = Image.open(live_ev_file).convert("RGB")
                        st.image(live_pil, width=200)
                        if st.button("Submit Live Image Evidence"):
                            ev_to_submit = EvidenceItem(
                                evidence_type=best_cand,
                                modality="image",
                                data_path_or_content=live_pil,
                                cost=1.0
                            )
                else:
                    text_input = st.text_area("Field Symptom Description / Context:", value="Dark brown irregular lesion margins with yellowing halo.")
                    if st.button("Submit Textual Evidence"):
                        ev_to_submit = EvidenceItem(
                            evidence_type=best_cand,
                            modality="text",
                            data_path_or_content=text_input,
                            cost=0.5
                        )

                if ev_to_submit is not None:
                    with st.spinner("Analyzing reliability, contradiction, and updating beliefs..."):
                        updated_state = pipeline.process_single_evidence(state, best_cand, ev_to_submit)
                        st.session_state["current_state"] = updated_state
                        st.rerun()

# 4. Diagnostic Evolution & Verified Audit Trail
if state and len(state["evidence_trail"]) > 0:
    st.write("---")
    st.subheader("4. Self-Verifying Audit Trail & Trajectory")
    
    col_t1, col_t2 = st.columns([1.5, 1])
    
    with col_t1:
        for item in state["evidence_trail"]:
            act = item.get("action", "ACCEPT")
            badge = f'<span class="badge-accept">ACCEPT</span>' if act == "ACCEPT" else (f'<span class="badge-downweight">DOWNWEIGHT</span>' if act == "DOWNWEIGHT" else f'<span class="badge-reject">REJECT</span>')
            
            with st.expander(f"Step {item['step']}: Acquired '{item['evidence_type'].replace('_', ' ').title()}' — Action: {act}", expanded=True):
                st.markdown(f"**Reliability Check:** {badge} | Quality Score: `{item['quality_score']:.2f}`", unsafe_allow_html=True)
                if item.get("reasons"):
                    st.write(f"- Quality Diagnostics: {', '.join(item['reasons'])}")
                
                is_rejected = act in ["REJECT", "REJECTED"]
                if is_rejected:
                    st.error("❌ **Observation Blocked & Discarded:** Image quality is unacceptable. The observation was rejected by Module 7 (Reliability Gating) and blocked from corrupting the diagnostic state. Previous beliefs were safely preserved.")
                else:
                    is_contra = item.get("contradiction", False)
                    if is_contra:
                        st.warning(f"⚠️ **Contradiction Detected:** {item.get('contradiction_explanation')}")
                    else:
                        st.success("✅ **Consistent:** Observation aligned with planned counterfactual trajectory.")
                
                st.write(f"**Updated Belief:** `{item['leading_after'].replace('_', ' ').title()}` ({item['leading_prob_after']*100:.1f}%) | **Entropy:** `{item['entropy_after']:.3f} bits`")


    with col_t2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### 🏆 Final Verified Diagnosis")
        st.markdown(f"**Disease:** `{state['leading_disease'].replace('_', ' ').title()}`")
        st.markdown(f"**Final Confidence:** `{state['leading_prob']*100:.1f}%`")
        st.markdown(f"**Total Steps Taken:** `{len(state['evidence_trail'])}`")
        
        init_lead_idx = int(np.argmax(state["initial_probs"]))
        init_lead_name = classes[init_lead_idx]
        
        # Calculate real entropy reduction ΔH = H0 - Ht
        init_p = np.array(state["initial_probs"])
        h0 = float(-np.sum(init_p * np.log2(init_p + 1e-12)))
        ht = float(state["entropy"])
        delta_h = max(0.0, h0 - ht)
        
        st.markdown(f"**Diagnostic Shift:** *{init_lead_name.replace('_', ' ').title()}* ➔ **{state['leading_disease'].replace('_', ' ').title()}**")
        st.markdown(f"**Current Entropy:** `{ht:.3f} bits`")
        st.markdown(f"**Uncertainty Reduction ($\Delta H$):** `{delta_h:.3f} bits`")
        st.markdown('</div>', unsafe_allow_html=True)
