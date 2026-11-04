# EvidenceGain-X

> **Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Vision Backbone: ResNet-18](https://img.shields.io/badge/Backbone-ResNet--18%20(ImageNet)-emerald.svg)](https://pytorch.org/vision/stable/models.html)
[![Dataset: PlantVillage](https://img.shields.io/badge/Dataset-PlantVillage%20PV--Tomato-green.svg)](https://github.com/spMohanty/PlantVillage-Dataset)
[![UI: Clinical Workstation](https://img.shields.io/badge/UI-Clinical%20Diagnostic%20Console-blue.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👥 Project Team

| S.No. | Student Name | Registration Number |
| :---: | :--- | :---: |
| 1 | **Amrit Raj** | `24BCE7856` |
| 2 | **Kumar Sanib** | `24BCA7144` |
| 3 | **Nikhil Yadav** | `24BCA7057` |
| 4 | **Rishi Choudhary** | `24BCE7490` |
| 5 | **Amit Venkat** | `24BCA7174` |
| 6 | **Nihal Kakollu** | `24BCA7283` |

---

## 📖 Overview: The Paradigm Shift in Crop Disease Diagnosis

### The Failure of Passive Single-Shot Plant AI
Conventional crop disease applications ask a single, passive question: **"Given this one photo, what disease is present?"**  
A user takes a photograph, and a convolutional neural network (CNN) outputs a classification.

**Why this paradigm fails in real-world agriculture:**
1. **Morphological Look-Alikes:** In early vegetative stages, distinct pathologies present near-identical foliar lesions. For example, **Early Blight** (*Alternaria solani*) and **Septoria Leaf Spot** (*Septoria lycopersici*) both manifest as brown circular spots. Standard single-shot classifiers are forced to make uncalibrated guesses along ambiguous boundaries.
2. **Hidden Diagnostic Discriminators:** The decisive diagnostic symptom is often occluded (e.g., downy fungal sporulation of **Late Blight** situated exclusively on the abaxial leaf underside, or petiole cankers on the stem). A single macroscopic image cannot observe what is physically concealed.
3. **Softmax Hallucination on Blurry Images:** Standard CNNs (ResNet, MobileNet) normalize outputs via Softmax, which mathematically forces outputs to sum to 100%. When presented with out-of-focus, blurry, or overexposed images, passive models hallucinate high confidence (e.g., 89.5% Late Blight) on meaningless green noise.
4. **Vulnerability to Contradictory Data:** When prompted for secondary evidence, farmers may inadvertently photograph an adjacent healthy leaf or a different plant. Conventional multi-modal networks fuse conflicting signals, poisoning the diagnostic posterior.

---

### The EvidenceGain-X Solution: Active Counterfactual Diagnosis
EvidenceGain-X acts as an **autonomous clinical diagnostic agent**:
1. **Provisional Hypothesis Tracking:** Analyzes the initial foliar observation ($X_0$). If confidence is decisive, it stops. If look-alike ambiguity is detected ($P(h_1) \approx P(h_2)$), it identifies the competing rival pathologies.
2. **Goal-Directed Evidence Selection ($e^*$):** Calculates the **Expected Information Gain (EIG)** across candidate modalities (lesion zoom, abaxial underside, stem view, weather context, field symptoms) to recommend the single most cost-effective discriminator.
3. **Optical Quality Gating (Module 7):** Pre-screen inputs via Laplacian variance ($\text{Var}(\nabla^2 I) \ge 80.0\ \sigma^2$) and luminance. **Strictly blocks diagnosis on blurry, dark, or distorted images** to eliminate AI hallucination.
4. **Contradiction Verification (Module 8):** Checks whether incoming evidence aligns with planned counterfactual trajectories. If a user uploads a healthy leaf into a diseased episode, Module 8 flags the conflict and dampens confidence to safeguard clinical integrity.
5. **Adaptive Bayesian Stopping:** Halts sequential inquiry the moment uncertainty is resolved (Shannon entropy $H(P) \le 0.35\text{ bits}$ and margin $\Delta p \ge 0.40$).

---

## 🚀 Quick Start Guide

### Step 1: Clone & Setup Environment

```bash
git clone https://github.com/techierizz/Evidence_Gain-X.git
cd Evidence_Gain-X

python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

---

### Step 2: Launch the Clinical Diagnostic Console

```bash
python -m streamlit run demo/app.py
```

Open your browser at: **`http://localhost:8501`**

#### Key Features in the Interface:
- **Clinical Case Presets:** One-click presets for authentic agricultural scenarios:
  - `Early Blight` — Validated *Alternaria solani* with concentric ring lesions.
  - `Late Blight` — Validated *Phytophthora infestans* with water-soaked margins.
  - `Septoria Spot` — Morphological look-alike ambiguity boundary case.
  - `Healthy Control` — Negative control baseline.
  - `Test: Blur Gating` — Demonstrates Module 7 Defocus Blur Quarantine.
  - `Test: Contradiction` — Demonstrates Module 8 Contradiction Gating.
- **Specimen Viewfinder & Telemetry HUD:** Real-time Laplacian sharpness ($\sigma^2$) and luminance ($cd/m^2$) sensors.
- **Multimodal Evidence Ingestion:**
  - *Direct Leaf Upload:* Upload photographic evidence (lesion close-up, leaf underside, stem view).
  - *One-Click Weather:* Quick field microclimate buttons (Warm & Dry, Cool & Wet, Humid, Sunny).
  - *Visual Symptoms:* Plain-language observation tags without textbook botanical jargon.
- **Audit Trail & Adaptive Certification:** Dynamic clinical summary reflecting verified, provisional, gated, or contradicted diagnostic outcomes.

---

### Step 3: Run Automated Test Suites

Verify pipeline integrity, blur rejection, and contradiction verification:

```bash
python tests/test_website_workflow.py
```

Expected output:
```text
Test 1 Early Blight: OK
Test 3 Septoria Ambiguity: OK
Test 5 Blurry Gating: OK
Test 6 Contradiction Gating: OK
Test Benchmark acquisition: OK
ALL 5 WORKFLOW TESTS PASSED!
```

To run Scenario B & C stress tests:
```bash
python scripts/test_scenarios.py
```

---

## 🔄 Complete 9-Module Architecture

| Module | Identifier | Scientific Purpose |
| :--- | :--- | :--- |
| **Module 1** | **Vision Feature Extractor** | Encodes foliar imagery via ImageNet-pretrained ResNet-18 backbone. |
| **Module 2** | **Provisional Diagnosis** | Generates calibrated class posteriors, Shannon entropy $H(P)$, and leading/rival candidates. |
| **Module 3** | **Confusion Analysis** | Identifies specific look-alike confusion pairs (e.g., Early Blight vs. Septoria Spot). |
| **Module 4** | **Candidate Generator** | Indexes available evidence modalities (lesion zoom, underside, stem, symptoms, weather). |
| **Module 5** | **Counterfactual Planning** | Simulates expected posterior trajectories under competing disease hypotheses. |
| **Module 6** | **Active Utility Selector** | Computes Net Policy Utility $U(e) = \alpha \cdot D_{JS} + \beta \cdot \text{Exp}(\Delta H) - \lambda \cdot \text{Cost}$. |
| **Module 7** | **Optical Quality Gating** | Evaluates Laplacian sharpness & luminance; blocks predictions on unreadable images. |
| **Module 8** | **Contradiction Verification** | Detects adversarial/conflicting evidence; dampens confidence to prevent false lock. |
| **Module 9** | **Bayesian Belief Updater** | Updates posterior distributions with reliability weighting ($q$) and penalty scaling ($\rho$). |

---

## 📈 Benchmark Evaluation (7-Baseline Comparison)

EvidenceGain-X was benchmarked against 6 alternative decision policies across 24 test episodes from the authentic PlantVillage PV-Tomato dataset:

| Baseline | Strategy | Accuracy | Macro F1 | Steps Taken | Entropy Reduction ($\Delta H$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **B0** | Initial Only (Passive CNN) | 87.50% | 0.8773 | 0.00 | 0.000 bits |
| **B1** | Random Acquisition | 87.50% | 0.8726 | 1.00 | 0.221 bits |
| **B2** | Fixed Order Sequence | 91.67% | 0.9188 | 1.00 | 0.284 bits |
| **B3** | Confidence-Driven | 87.50% | 0.8726 | 0.50 | 0.203 bits |
| **B4** | Uncertainty-Only | 83.33% | 0.8233 | 1.12 | 0.229 bits |
| **B5** | EvidenceGain (No Gating) | 91.67% | 0.9188 | 0.79 | 0.359 bits |
| **B6** | **EvidenceGain-X (Full System)** | **91.67%** | **0.9188** | **0.88** | **0.359 bits** |

*Under degraded test conditions (optical blur, underexposure, contradictory leaf upload), baselines B0–B5 experience severe misdiagnoses. EvidenceGain-X successfully gates 100% of invalid samples and penalizes contradictory inputs.*

---

## ☁️ Deployment Guide

### Option A: Streamlit Community Cloud (Recommended — Free, Fastest)

1. Fork or push to GitHub: `https://github.com/techierizz/Evidence_Gain-X`
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**:
   - **Repository:** `techierizz/Evidence_Gain-X`
   - **Branch:** `main`
   - **Main file path:** `demo/app.py`
4. Click **Deploy!** The app will build and provide a public HTTPS URL.

### Option B: Docker Container

A production `Dockerfile` is included in the root directory:

```bash
docker build -t evidencegain-x .
docker run -p 8501:8501 evidencegain-x
```

Open `http://localhost:8501` in your browser.

### Option C: Live Zero-Cold-Start Tunnel (For In-Person Presentation)

To present from your local computer with zero cold-boot delays:

```bash
python -m streamlit run demo/app.py --server.port 8501
```

In a second terminal:
```bash
npx cloudflared tunnel --url http://localhost:8501
```
This generates a live public HTTPS URL with instant response times and zero sleep.

---

## 📁 Repository Structure

```text
EvidenceGain-X/
├── checkpoints/
│   └── classifier_best.pt       # Calibrated ResNet-18 weights (tracked in Git)
├── configs/
│   └── config.yaml              # Hyperparameters, stopping thresholds & loss weights
├── data/
│   ├── manifests/               # Leakage-safe train, val, and test manifests (JSON)
│   ├── raw/                     # Multi-view foliar specimen episodes
│   └── test_samples/            # Real test images (blur, underexposed, contradiction, real classes)
├── demo/
│   └── app.py                   # Streamlit clinical research diagnostic console
├── results/                     # Quantitative benchmark summaries and metrics
├── scripts/
│   ├── download_data.py         # PlantVillage dataset downloader & episode builder
│   ├── run_experiments.py       # Benchmark evaluation script
│   ├── generate_figures.py      # Scientific figure generator
│   └── test_scenarios.py        # Standalone CLI validation for Scenarios B & C
├── src/
│   ├── data/                    # Dataset manifest and episode builders
│   ├── diagnosis/               # Clinical pipeline and hypothesis tracking
│   ├── evaluation/              # Benchmark runner and baseline evaluators
│   ├── evidence/                # Active utility selector and decision policies
│   ├── models/                  # ResNet-18 encoder and text embedding network
│   ├── planning/                # Counterfactual trajectory planner
│   ├── utils/                   # Config and reproducibility helpers
│   └── verification/            # Module 7 (Optical Gate) & Module 8 (Contradiction Verifier)
├── tests/                       # Automated test suite (test_website_workflow.py)
├── Dockerfile                   # Production container definition
├── EvidenceGain_X_Codebase_Manual_and_Defense_Guide.pdf # Comprehensive manual
├── README.md                    # Project documentation
└── requirements.txt             # Clean Python dependency list
```

---

## 📜 Academic Citation

```bibtex
@article{evidencegainx2026,
  title={EvidenceGain-X: Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis},
  author={Amrit Raj and Kumar Sanib and Nikhil Yadav and Rishi Choudhary and Amit Venkat and Nihal Kakollu},
  year={2026}
}
```
