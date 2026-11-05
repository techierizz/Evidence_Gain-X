# EvidenceGain-X: Counterfactual, Self-Verifying Next-Best Evidence Acquisition for Crop Disease Diagnosis

**Rishi Choudhary**, *Student Member, IEEE*, **Kumar Sanib**, *Student Member, IEEE*, **Amrit Raj**, *Student Member, IEEE*, **Nikhil Yadav**, *Student Member, IEEE*, **Amit Venkat**, *Student Member, IEEE*, **Nihal Kakollu**, *Student Member, IEEE*, and **Devulapalli Sudheer**, *Member, IEEE*

*Department of Computer Science and Engineering, School of Computer Science and Engineering (SCOPE), VIT-AP University, Amaravati, Andhra Pradesh 522237, India*  
*(Corresponding Author Email: amritraj08888@gmail.com)*  
*Code Repository: https://github.com/techierizz/Evidence_Gain-X*

---

### **Abstract**
Deployed computer vision systems for agricultural plant pathology overwhelmingly operate under a passive, single-shot classification regime: an end-user submits a single foliar photograph, and a convolutional neural network outputs an uncalibrated point prediction. While this paradigm yields benchmark accuracies exceeding 95% on curated, homogeneous datasets, it fails catastrophically in real-world agronomic environments due to four persistent vulnerabilities: (i) acute morphological look-alike ambiguity along early vegetative decision boundaries (e.g., *Alternaria solani* vs. *Septoria lycopersici*), (ii) physical occlusion of pathognomonic discriminators (such as abaxial downy sporulation in *Phytophthora infestans*), (iii) unconstrained Softmax confidence hallucination on optically degraded specimens, and (iv) Bayesian posterior corruption caused by inadvertent user uploads of healthy or foreign vegetation.

To resolve these failure modes, this paper presents **EvidenceGain-X**, an autonomous, closed-loop diagnostic framework comprising nine functionally distinct modules that reformulates plant pathology from passive inference into an active, self-verifying clinical dialogue. EvidenceGain-X quantifies its predictive uncertainty through temperature-scaled Shannon entropy ($T = 1.8928$, optimized via L-BFGS on an isolated validation partition to reduce Expected Calibration Error by 78.2%), simulates counterfactual posterior belief trajectories under competing diagnostic hypotheses using an empirical discriminative affinity matrix, and determines the optimal next-best evidence query ($e^*$) by maximizing a cost-aware Net Policy Utility function across five multimodal modalities (lesion zoom, abaxial underside, stem petiole, field symptoms, and microclimate context). Crucially, the architecture incorporates two cyber-physical safety shields: **Module 7 Optical Quality Gating** (discrete Laplacian variance threshold $\sigma^2 \ge 80.0$ and luminance bounds $[30, 230]\text{ cd/m}^2$) enforcing deterministic blur/glare interception on both initial and follow-up inputs, and **Module 8 Contradiction & Specimen Mismatch Guard** (directional cosine divergence threshold $\ge 0.40$ and out-of-distribution prior-evidence bounds), which quarantines incongruent foliage.

Evaluated on 24 multi-view diagnostic episodes from the PlantVillage tomato benchmark under leakage-free, plant-grouped partitioning (70/15/15), EvidenceGain-X elevates diagnostic accuracy from 87.50% (passive single-shot) to 91.67% and attains a Macro F1 score of 0.9188 in an average of only 0.88 sequential observation steps, while achieving the highest Shannon entropy reduction (0.359 bits) among all evaluated policies. Under adversarial and noisy conditions, EvidenceGain-X achieves a 100% rejection rate against motion-blurred, underexposed, and cross-specimen contaminant inputs—a resilience completely unobserved in existing baselines. The entire pipeline operates with an edge CPU latency under 30 ms and a memory footprint of 44.5 MB, establishing a production standard for trustworthy edge AI in digital agriculture.

**Index Terms**—Active sequential computer vision, Bayesian belief updating, counterfactual planning, crop disease diagnosis, foliar pathology, optical quality gating, precision agriculture, self-verifying artificial intelligence, temperature scaling, uncertainty quantification.

---

## I. INTRODUCTION

### A. Agronomic Context and the Crisis of Passive Vision AI
Foliar crop pathologies account for an estimated 20% to 40% reduction in global crop yields annually, imperiling food security and economic stability for over 500 million smallholder farming households across South Asia and sub-Saharan Africa [1], [2]. Among staple horticultural commodities, tomato (*Solanum lycopersicum*) production exceeds 180 million metric tonnes annually; however, the crop is exceptionally susceptible to aggressive fungal and oomycete pathogens, primarily Early Blight (*Alternaria solani*), Late Blight (*Phytophthora infestans*), and Septoria Leaf Spot (*Septoria lycopersici*) [3]. In particular, *Phytophthora infestans* exhibits explosive infection dynamics, capable of transforming a healthy canopy into completely necrotic tissue within 7 to 10 days under humid, temperate microclimates. The asymmetric economic hazard of diagnostic error—where a false negative for Late Blight results in catastrophic harvest liquidation—demands diagnostic systems that exhibit uncompromising clinical reliability [4].

Over the past decade, deep learning architectures—most notably deep Convolutional Neural Networks (CNNs) and Vision Transformers—have achieved reported classification accuracies exceeding 98% on curated agricultural benchmark repositories like PlantVillage [1], [5]. Despite these high benchmark scores, practical field deployment of mobile agricultural diagnostic tools remains plagued by severe misdiagnoses. Our empirical investigation reveals four structural failure modes intrinsic to the conventional **passive single-shot paradigm**:
1. **Morphological Look-Alike Ambiguity:** In early infection stages, distinct fungal taxa converge morphologically. *Alternaria solani* and *Septoria lycopersici* both manifest as small, circular, brown necrotic flecks encircled by chlorotic yellow halos. In top-down macroscopic photography, spatial sensor resolution is fundamentally insufficient to resolve microscopic pycnidia (black fungal fruiting bodies diagnostic of Septoria) from immature concentric target rings. Standard classifiers force an uncalibrated, high-entropy guess between bimodal posteriors ($P \approx 50\%$ vs. $50\%$).
2. **Observational Occlusion of Diagnostic Discriminators:** Plant pathologies are intrinsically three-dimensional, anatomical phenomena. Pathognomonic symptoms frequently develop on leaf structures concealed from conventional overhead viewpoints. Specifically, the defining morphological proof of Late Blight is delicate, white downy sporulation located exclusively on the **abaxial (underside) leaf lamina**, or dark petiole lesions along the stem. A single macroscopic photograph cannot inspect what is physically occluded.
3. **Softmax Hallucination on Degraded Imagery:** Standard neural network heads apply the Softmax activation function over final classification logits $\mathbf{z}$:
   $$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
   Because Softmax mathematically normalizes logits to sum to unity ($\sum \sigma(\mathbf{z})_i = 1$), it lacks an explicit "uncertain" or "corrupted input" state. When a farmer captures an out-of-focus, motion-blurred, or severely underexposed photograph, the convolutional feature extractor maps low-frequency green blur to arbitrary high-confidence predictions (e.g., assigning 89.5% probability of Late Blight to empty blur), creating severe safety risks.
4. **Cross-Specimen Contamination & Contradiction:** In sequential multimodal inquiry, non-expert users frequently photograph an adjacent asymptomatic leaf, capture a weed, or ingest an unrelated specimen. Conventional feature-averaging or Bayesian fusion architectures blindly absorb contradictory inputs, corrupting prior diagnostic beliefs and destabilizing the clinical decision path.

### B. The EvidenceGain-X Paradigm: Active Clinical Vision
To overcome these limitations, this paper introduces **EvidenceGain-X**, an autonomous diagnostic system modeled upon human clinical reasoning. When an expert plant pathologist encounters an ambiguous foliar presentation, they do not commit to an immediate guess. Instead, they quantify diagnostic uncertainty, formulate competing differential hypotheses ($h_1$ vs. $h_2$), simulate what information prospective physical examinations will yield, and instruct the farmer to perform a specific targeted maneuver (e.g., *"turn the leaf over to inspect the underside"*). Furthermore, incoming evidence is rigorously vetted for photographic clarity and specimen congruence before clinical belief is updated.

EvidenceGain-X formalizes this clinical methodology into a nine-module closed-loop pipeline. By integrating temperature-scaled uncertainty gating, counterfactual trajectory simulation, multi-objective utility optimization, optical blur gating, and directional contradiction verification, EvidenceGain-X transforms crop pathology from passive single-shot inference into an active, self-correcting clinical dialogue.

### C. Summary of Contributions
The primary scientific contributions of this work are as follows:
1. **Counterfactual Evidence Planning (Module 5):** A goal-directed planning engine that models hypothetical posterior trajectories under competing disease hypotheses, employing a domain-specific discriminative affinity matrix ($\mathbf{A} \in \mathbb{R}^{M \times K \times K}$) calibrated from agronomic literature to compute expected information gain without requiring intractable Monte Carlo approximations.
2. **Cost-Aware Multi-Factor Utility Function (Module 6):** A multi-objective query selection policy combining five Pareto hyperparameters ($\alpha_{\text{sep}} = 1.0$, $\beta_{\text{unc}} = 0.8$, $\gamma_{\text{diag}} = 1.2$, $\lambda_{\text{cost}} = 0.3$, $\mu_{\text{risk}} = 0.5$) that balances expected entropy reduction against physical farmer acquisition effort and operational failure risk.
3. **Dual Cyber-Physical Safety Shields (Modules 7 and 8):**
   - *Module 7 (Optical Quality Gating):* Deterministic pre-screening enforcing Laplacian edge variance ($\sigma^2 \ge 80.0$) and luminance bounds ($[30, 230]\text{ cd/m}^2$) on **both initial specimen ingestion ($X_0$) and sequential evidence**, eliminating Softmax hallucination on degraded imagery.
   - *Module 8 (Contradiction & Specimen Mismatch Guard):* Dual-criterion verification that detects directional cosine divergence ($S_{\text{contra}} \ge 0.40$) and foreign pathology intrusion ($\theta_{\text{ev}} \notin \{h_1, h_2\}$ with prior $< 0.05$ and evidence $\ge 0.35$), quarantining corrupted uploads and preserving prior diagnostic states.
4. **Leakage-Safe Validation & Open-Source Implementation:** Complete empirical validation across 24 PlantVillage episodes partitioned under rigorous physical plant-level isolation (zero inter-partition leaf leakage), accompanied by an open-source, edge-deployable implementation with a 44.5 MB footprint and sub-30 ms CPU inference latency.

---

## II. RELATED WORK

### A. Deep Learning and Foliar Pathology Benchmarks
The emergence of deep learning in precision agriculture was catalyzed by the PlantVillage repository [1], [2], comprising 54,306 curated photographs across 26 plant diseases. Early studies applying AlexNet, VGG, and Inception architectures demonstrated classification accuracies above 99% [2]. Subsequent works incorporated residual learning (ResNet [5]), dense connectivity (DenseNet), and Vision Transformers (ViT). However, Ferentinos [4] demonstrated that random train-test splitting in foliar datasets introduces severe **data leakage**: multiple photos of the same physical leaf or plant canopy appear simultaneously across training and evaluation partitions. Consequently, models memorize background greenhouse artifacts and camera idiosyncrasies rather than learning true morphological pathology. Plant-grouped partitioning, as enforced in this study, provides an accurate reflection of generalization performance by ensuring zero physical plant overlap.

Furthermore, existing literature almost universally treats disease identification as a passive, single-shot mapping $f: \mathcal{X} \to \mathcal{Y}$ [2], [4]. Ramcharan *et al.* [6] showed that single-shot CNN models trained on clean lab images suffer severe degradation when exposed to unconstrained field imagery. EvidenceGain-X addresses this foundational limitation by replacing single-shot guessing with active sequential acquisition.

### B. Uncertainty Quantification and Model Calibration
Modern deep neural networks trained with cross-entropy loss suffer from systematic overconfidence, generating peaky softmax distributions that do not reflect true posterior probability [7]. Guo *et al.* [7] established that post-hoc **temperature scaling** effectively calibrates neural probability distributions without altering argmax classification rankings. While Bayesian neural networks (e.g., Monte Carlo Dropout [8]) and Deep Ensembles [9] estimate epistemic uncertainty, they incur significant computational overhead: MC Dropout requires $N$ stochastic forward passes, while ensembles scale model footprint and memory requirements linearly by $M \times$. In resource-constrained agricultural edge contexts, temperature scaling represents the optimal calibration mechanism, adding zero runtime parameter memory or inference delay.

### C. Active Inquiry and Bayesian Experimental Design
Active machine learning traditionally focuses on pool-based sample selection to minimize annotation labelling costs during model training [10]. EvidenceGain-X adapts this principle to test-time sequential inference, drawing upon **Bayesian Optimal Experimental Design** [11], [12]. In our framework, the goal is not to label training samples, but to sequentially acquire complementary physical modalities (lesion zoom, abaxial underside, microclimate) for a single diagnostic subject until uncertainty is resolved below clinical safety thresholds.

---

## III. THE PROPOSED EVIDENCEGAIN-X FRAMEWORK

The complete architecture of EvidenceGain-X operates as a closed-loop active inference system organized into nine functionally cohesive modules, illustrated in Fig. 1.

```
                      +----------------------------------------+
                      |   Initial Specimen Photograph (X0)     |
                      +----------------------------------------+
                                          |
                                          v
                      +----------------------------------------+
                      | Module 7: Optical Quality Gating (HUD) |
                      |    Var(Laplacian) >= 80.0, Brightness  |
                      +----------------------------------------+
                               |                      |
                        [REJECT/BLUR]            [PASS/CLEAN]
                               |                      |
                               v                      v
                      +----------------+   +------------------------------------+
                      | Gated / Quaran.|   | Module 1: ResNet-18 Vision Encoder |
                      | Zero Prob Hall.|   |   h0 = f_phi(X0) in R^512          |
                      +----------------+   +------------------------------------+
                                                              |
                                                              v
                                           +------------------------------------+
                                           | Module 2: Platt Temp Scaling (M2)  |
                                           |    P(theta_c) = softmax(z_c / T)   |
                                           |    T = 1.8928, H(P), Margin Delta p|
                                           +------------------------------------+
                                                              |
                                                              v
                                           +------------------------------------+
                                           |      Ambiguity Decision Gate       |
                                           |   H(P) <= 0.35 & Delta p >= 0.70?  |
                                           +------------------------------------+
                                                    |                     |
                                                 [YES]                  [NO]
                                                    |                     |
                                                    v                     v
                                           +----------------+   +------------------------------------+
                                           | Definite Stop  |   | Module 3: Confusion Analyzer (M3)  |
                                           | Certified Lead |   |   Identify Look-Alikes (H1 vs H2)  |
                                           +----------------+   +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 4: Candidate Modalities     |
                                                                |   E = {Zoom, Abaxial, Stem, ...}   |
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 5: Counterfactual Planner   |
                                                                |   Simulate P(theta | e, H1/H2)     |
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 6: Net Policy Utility (M6)  |
                                                                |   Argmax U(e) = a*JSD + b*EIG - ...|
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Target Clue Query Dispatched (e*)  |
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 7: Evidence Optical Gate    |
                                                                |   Pre-screen secondary input       |
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 8: Specimen Mismatch Guard  |
                                                                |   Cosine contra & prior boundary   |
                                                                +------------------------------------+
                                                                                  |
                                                                                  v
                                                                +------------------------------------+
                                                                | Module 9: Bayesian Belief Updater  |
                                                                |   P_new ~ P_prior * [L(e)]^(q*rho) |
                                                                +------------------------------------+
                                                                                  |
                                                                                  +---- (Iterate Loop)
```
*Fig. 1. End-to-end operational signal path of the 9-module EvidenceGain-X framework.*

---

### A. Module 1: Vision Backbone Encoder
Given a macroscopic leaf observation $X_0 \in \mathbb{R}^{H \times W \times 3}$, preprocessing applies deterministic bicubic scaling to $224 \times 224$ pixels, followed by ImageNet channel mean subtraction ($\mu = [0.485, 0.456, 0.406]$) and standard deviation scaling ($\sigma = [0.229, 0.224, 0.225]$). Feature extraction is performed via an ImageNet-pretrained ResNet-18 architecture [5]:
$$\mathbf{h}_0 = f_{\phi}(X_0) \in \mathbb{R}^{512}$$
A classification head $\mathbf{W}_{\text{cls}} \in \mathbb{R}^{4 \times 512}$ with bias $\mathbf{b}_{\text{cls}} \in \mathbb{R}^4$ maps the latent feature vector $\mathbf{h}_0$ to unnormalized diagnostic logits:
$$\mathbf{z} = \mathbf{W}_{\text{cls}} \mathbf{h}_0 + \mathbf{b}_{\text{cls}}$$

ResNet-18 was selected over heavier backbones (ResNet-50, ViT-B/16) based on empirical edge execution constraints: at 44.0 MB parameters and 12 ms CPU forward-pass latency, it delivers real-time inference on edge processors (such as ARM Cortex-A72 in Raspberry Pi 4 and mobile microcontrollers) while preventing overfitting on specialized agricultural subsets.

### B. Module 2: Temperature-Scaled Uncertainty Quantification
To correct deep network overconfidence, raw classification logits are calibrated via post-hoc **Platt Temperature Scaling** [7]:
$$P(\theta_c \mid X_0) = \frac{\exp(z_c / T)}{\sum_{j=1}^K \exp(z_j / T)} \quad \text{--- (1)}$$
where $K = 4$ denotes the target pathology taxonomy: $\Theta = \{\text{Early Blight}, \text{Late Blight}, \text{Septoria Leaf Spot}, \text{Healthy}\}$.

The optimal temperature parameter $T^* = 1.8928$ is learned by minimizing negative log-likelihood over the uncorrupted validation partition using L-BFGS:
$$T^* = \operatorname*{arg\,min}_{T} -\sum_{n=1}^{N_{\text{val}}} \log \left( \frac{\exp(z_{y_n} / T)}{\sum_{j=1}^K \exp(z_j / T)} \right)$$
All underlying network weights $\phi$ and $\mathbf{W}_{\text{cls}}$ remain frozen. Because $T^* > 1.0$, the calibration properly softens overly confident logit distributions, ensuring that computed entropy values accurately reflect empirical uncertainty.

From the calibrated posterior distribution $P(\Theta)$, Module 2 computes three vital diagnostic scalars:
1. **Shannon Entropy (in bits):**
   $$H(P) = -\sum_{c=1}^K P(\theta_c) \log_2 P(\theta_c) \quad \text{--- (2)}$$
   For $K=4$, maximum uniform entropy equals $\log_2(4) = 2.0\text{ bits}$; complete deterministic certainty equals $0.0\text{ bits}$.
2. **Leading & Rival Hypotheses:**
   $$h_1 = \operatorname*{arg\,max}_{\theta_c \in \Theta} P(\theta_c), \quad h_2 = \operatorname*{arg\,max}_{\theta_c \in \Theta \setminus \{h_1\}} P(\theta_c)$$
3. **Decision Margin:**
   $$\Delta p = P(h_1) - P(h_2)$$

### C. Ambiguity Decision Gate & Stopping Criteria
The ambiguity gate assesses whether the current diagnostic belief warrants immediate clinical termination or requires active sequential inquiry. The stopping condition is defined by the joint conjunction:
$$\text{TERMINATE} \iff \left[ H(P) \le \tau_H \right] \;\wedge\; \left[ \Delta p \ge \tau_m \right] \quad \text{--- (3)}$$
where $\tau_H = 0.35\text{ bits}$ and $\tau_m = 0.70$. If both conditions hold simultaneously, the system commits to diagnosis $h_1$ at zero additional inquiry cost. If either condition is violated, the diagnosis is declared **ambiguous**, and control transfers to the active inquiry engine.

### D. Module 3: Disease-Confusion Analysis
When ambiguity is declared, Module 3 inspects probability mass concentrations across the rival pair $(h_1, h_2)$ to classify the specific clinical confusion typology:
- **Systemic Blight Confusion:** $P(\text{Early Blight}) + P(\text{Late Blight}) \ge 0.75$, indicating macroscopic foliar necrosis where pathogen etiology must be determined.
- **Foliar Spot Confusion:** $P(\text{Early Blight}) + P(\text{Septoria}) \ge 0.75$, representing near-identical circular lesions requiring microscopic structural discrimination (pycnidia vs. concentric rings).

### E. Module 4: Candidate Evidence Pool and Agronomic Cost Modeling
EvidenceGain-X maintains an operational catalog of five candidate multimodal evidence modalities $\mathcal{E}$, parameterized by acquisition cost $C(e)$ and physical capture failure risk $R(e)$:
1. **Lesion Close-Up ($e_1$):** High-magnification macro zoom ($C = 1.0, R = 0.20$).
2. **Abaxial Leaf Underside ($e_2$):** Inverted leaf photograph revealing downy sporulation ($C = 1.2, R = 0.30$).
3. **Stem Petiole View ($e_3$):** Stem photograph inspecting vascular cankers ($C = 1.5, R = 0.35$).
4. **Field Symptom Description ($e_4$):** Textual observation encoded via TextEvidenceEncoder ($C = 0.4, R = 0.10$).
5. **Microclimate Context ($e_5$):** Temperature, relative humidity, and rainfall context ($C = 0.4, R = 0.15$).

Acquired modalities are removed dynamically from $\mathcal{E}$ to prevent redundant querying. A maximum horizon budget $B = 3$ prevents unbounded farmer burden.

### F. Module 5: Counterfactual Evidence Planning
Rather than selecting evidence reactively, Module 5 simulates prospective diagnostic futures under competing hypotheses $h_1$ and $h_2$. For each available clue $e \in \mathcal{E}$, the planner simulates two hypothetical posterior distributions: $P(h \mid e, h_1)$ (if evidence confirms $h_1$) and $P(h \mid e, h_2)$ (if evidence confirms $h_2$).

Simulations are governed by an empirical **Discriminative Affinity Matrix** $\mathbf{A} \in \mathbb{R}^{|\mathcal{E}| \times K \times K}$, where entry $A[e, i, j] \in [0, 1]$ represents the domain-specific power of modality $e$ to separate pathology $i$ from pathology $j$. Under the counterfactual assumption that $e$ supports $h_1$, probability mass is transferred:
$$\Delta P = s \cdot \kappa \cdot P(h_2), \quad \text{where } s = A[e, h_1, h_2]$$
with conservative scaling $\kappa = 0.7$ to prevent over-optimistic simulation shifts.

The divergence between competing simulated futures is measured via the symmetric **Jensen-Shannon Divergence (JSD)**:
$$D_{\text{JS}}(P_1 \parallel P_2) = \frac{1}{2} D_{\text{KL}}(P_1 \parallel M) + \frac{1}{2} D_{\text{KL}}(P_2 \parallel M) \quad \text{--- (4)}$$
where $M = \frac{1}{2}(P_1 + P_2)$, and $D_{\text{KL}}(P \parallel Q) = \sum_c P(c) \log_2 \frac{P(c)}{Q(c)}$. Modalities with high $D_{\text{JS}}$ possess strong counterfactual decisiveness, guaranteeing diagnostic disambiguation regardless of which disease is true.

### G. Module 6: Multi-Factor Utility Scoring
The optimal query $e^*$ is chosen by maximizing the composite Net Policy Utility function $U(e)$:
$$e^* = \operatorname*{arg\,max}_{e \in \mathcal{E}} \left[ \alpha \cdot D_{\text{JS}}(P_1 \parallel P_2) + \beta \cdot \mathbb{E}[\Delta H(e)] + \gamma \cdot \mathcal{S}(e) - \lambda \cdot C(e) - \mu \cdot R(e) \right] \quad \text{--- (5)}$$
where:
- $\mathbb{E}[\Delta H(e)]$ is predicted Expected Information Gain generated by the lightweight UtilityScoringNetwork MLP.
- $\mathcal{S}(e) = A[e, h_1, h_2]$ is the direct pairwise separation power.
- $C(e)$ and $R(e)$ are modality physical acquisition cost and failure risk.
- Calibrated Pareto hyperparameters: $\alpha = 1.0$, $\beta = 0.8$, $\gamma = 1.2$, $\lambda = 0.3$, and $\mu = 0.5$.

### H. Module 7: Optical Quality Gating (Safety Shield 1)
To permanently eliminate Softmax hallucination on degraded imagery, **Module 7 intercepts both initial photographs ($X_0$) and follow-up photographic clues** prior to neural network ingestion.
1. **Laplacian Blur Assessment:** Image sharpness is computed via the variance of the 2D discrete Laplacian convolution:
   $$\sigma^2(I) = \frac{1}{N} \sum_{x,y} \left( \nabla^2 I_{\text{gray}}(x,y) - \mu_{\nabla} \right)^2 \quad \text{--- (6)}$$
   $$\text{where } \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} = I * \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$
2. **Luminance Histogram Assessment:** Mean illumination is computed as $\mu_L = \frac{1}{N} \sum_{x,y} I_{\text{gray}}(x,y)$.

A composite optical quality score $q \in [0, 1]$ is generated:
$$q = \min\left(1.0, \frac{\sigma^2(I)}{1.5 \cdot \tau_{\text{blur}}}\right) \cdot f_{\text{bright}}(\mu_L)$$
where $\tau_{\text{blur}} = 80.0\ \sigma^2$, and $f_{\text{bright}}(\mu_L) = 1.0$ for $\mu_L \in [30, 230]\text{ cd/m}^2$ (ramping linearly to 0 outside).
- If $\sigma^2(I) < 80.0$ or $\mu_L \notin [30, 230]$: **ACTION = REJECT** ($q < 0.35$). The image is quarantined, model forward passes are bypassed, and the user is prompted to retake the photo without emitting erroneous disease probabilities.
- If $q \ge 0.70$ and $\sigma^2(I) \ge 80.0$: **ACTION = ACCEPT**.

### I. Module 8: Contradiction Verification and Specimen Mismatch Guard (Safety Shield 2)
Module 8 protects the Bayesian posterior from corrupted, contradictory, or cross-specimen contaminant uploads through two distinct verification routines:
1. **Directional Trajectory Contradiction:**
   Let $\Delta_{\text{exp}, 1} = P(h \mid e, h_1) - P_{\text{prior}}$ and $\Delta_{\text{exp}, 2} = P(h \mid e, h_2) - P_{\text{prior}}$ denote the planned counterfactual shift vectors, and $\Delta_{\text{obs}} = P_{\text{obs}} - P_{\text{prior}}$ denote the observed belief shift. The directional cosine alignment is:
   $$\cos_k = \frac{\Delta_{\text{obs}} \cdot \Delta_{\text{exp}, k}}{\|\Delta_{\text{obs}}\| \|\Delta_{\text{exp}, k}\|}, \quad S_{\text{contra}} = \frac{1 - \max(\cos_1, \cos_2)}{2} \quad \text{--- (7)}$$
   If $S_{\text{contra}} \ge 0.40$, the observation is flagged as contradictory, triggering a confidence dampening penalty:
   $$\rho = 1.0 - 0.50 \cdot S_{\text{contra}}$$
2. **Specimen Mismatch Guard:**
   If incoming evidence exhibits high likelihood for an alien pathology $\theta_{\text{ev}}$ that was excluded during initial observation:
   $$\theta_{\text{ev}} \notin \{h_1, h_2\} \;\wedge\; P_{\text{prior}}(\theta_{\text{ev}}) < 0.05 \;\wedge\; P_{\text{ev}}(\theta_{\text{ev}}) \ge 0.35$$
   the system detects a **Specimen Mismatch** (e.g., uploading a healthy leaf into an active blight inquiry). The evidence is completely **QUARANTINED** ($\rho = 0.0$), halting posterior corruption while retaining prior clinical beliefs.

### J. Module 9: Reliability-Weighted Bayesian Belief Updater
Valid evidence updates the diagnostic posterior according to the modified Bayesian update rule:
$$P_{\text{new}}(\theta_c \mid e) = \frac{P_{\text{prior}}(\theta_c) \cdot \left[ L(e \mid \theta_c) \right]^\gamma}{\sum_{j=1}^K P_{\text{prior}}(\theta_j) \cdot \left[ L(e \mid \theta_j) \right]^\gamma} \quad \text{--- (8)}$$
where $L(e \mid \theta_c)$ is the multimodal evidence likelihood, and the effective evidence exponent $\gamma$ dynamically scales with optical quality $q$ and contradiction factor $\rho$:
$$\gamma = \operatorname{clip}(q \cdot \rho, 0.05, 1.5)$$
On pristine congruent evidence ($q=1.0, \rho=1.0$), $\gamma = 1.0$ (exact Bayesian updating). On degraded inputs ($q < 0.7$), $\gamma$ attenuates belief shifts. On quarantined mismatches ($\rho = 0.0$), $\gamma \to 0.05$, neutralizing the contaminant. Following belief updating, control returns to the Ambiguity Decision Gate for re-evaluation.

---

## IV. SYSTEM IMPLEMENTATION & HARDWARE ARCHITECTURE

### A. Software Stack and Modular Implementation
EvidenceGain-X is implemented in Python 3.10+ using PyTorch 2.2+, torchvision 0.17+, OpenCV-headless 4.8+, SciPy 1.11+, and Streamlit 1.30+. The architecture strictly decouples algorithmic modules into isolated subpackages (`src/models/`, `src/planning/`, `src/verification/`, `src/diagnosis/`), ensuring independent testability and maintainability.

### B. Multimodal Text Evidence Encoder
To project textual field observations into the visual latent space, we designed a lightweight `TextEvidenceEncoder`. It maps a curated 35-term agricultural vocabulary covering lesion morphology (*concentric*, *rings*, *halo*, *pycnidia*), texture (*water-soaked*, *velvety*), and weather (*humid*, *heavy dew*) into a 64-dimensional embedding, followed by a projection head: $\text{Linear}(64, 128) \to \text{ReLU} \to \text{Linear}(128, 512) \to \text{LayerNorm}(512)$. Out-of-vocabulary tokens are hashed deterministically. LayerNorm aligns text embedding norms with ResNet-18 visual feature vectors ($\|\mathbf{h}\|_2$), enabling stable cosine distance comparisons.

---

## V. EXPERIMENTAL SETUP & METHODOLOGY

### A. Benchmark Dataset and Leakage-Free Partitioning
Experiments were conducted using 200 authentic multi-modal episodes derived from the PlantVillage tomato corpus [1]. To eliminate data leakage [4], partitioning was performed by physical plant identity:
- **Training Set (70%):** 140 episodes (strictly plant-isolated).
- **Validation Set (15%):** 30 episodes (used for Platt calibration and hyperparameter tuning).
- **Testing Benchmark (15%):** 24 fully validated multi-view episodes across the four target pathologies.

Zero physical plant overlap exists across partitions, ensuring true out-of-sample generalization.

### B. Evaluated Baseline Diagnostic Policies
We benchmark EvidenceGain-X against six competitive decision policies:
- **B0 (Initial Only):** Conventional passive single-shot ResNet-18.
- **B1 (Random Policy):** Acquires follow-up observations uniformly at random.
- **B2 (Fixed Order Sequence):** Rigid deterministic sequence ($e_1 \to e_2 \to e_3 \to e_4 \to e_5$).
- **B3 (Confidence-Driven):** Queries randomly until maximum probability exceeds 0.75.
- **B4 (Uncertainty-Only):** Queries modalities with highest generic historical entropy reduction without considering the rival pair.
- **B5 (EvidenceGain - Ablated):** Full counterfactual utility optimization without Module 7 & 8 safety shields.
- **B6 (EvidenceGain-X - Proposed):** Complete nine-module framework with dual safety shields.

---

## VI. RESULTS AND QUANTITATIVE ANALYSIS

### A. Diagnostic Performance Across Benchmark Episodes
Table I summarizes quantitative performance across all seven policies on the 24-case benchmark.

```
TABLE I
QUANTITATIVE BENCHMARK EVALUATION (24 PLANT-GROUPED TOMATO EPISODES)

Policy                  Strategy Description        Accuracy   Macro F1   Mean Steps   Entropy Delta (bits)   Adversarial Robustness
--------------------------------------------------------------------------------------------------------------------------------------
B0: Initial Only        Passive Single-Shot CNN      87.50%     0.8773       0.00             0.000             0.0% (Hallucinates)
B1: Random Policy       Uniform Random Selection     87.50%     0.8726       1.00             0.221            20.0% (Vulnerable)
B2: Fixed Sequence      Static Predefined Order      91.67%     0.9188       1.00             0.284            30.0% (Rigid)
B3: Confidence-Driven   Greedy Max Probability       87.50%     0.8726       0.50             0.203            15.0% (Overconfident)
B4: Uncertainty-Only    Greedy Minimum Entropy       83.33%     0.8233       1.12             0.229            25.0% (Drifts)
B5: EvidenceGain        EIG Utility (No Safety)      91.67%     0.9188       0.79             0.359            40.0% (Poisoned)
B6: EvidenceGain-X      Full Proposed Framework      91.67%     0.9188       0.88             0.359           100.0% (Gated / Safe)
--------------------------------------------------------------------------------------------------------------------------------------
```

Passive single-shot classification (B0) correctly diagnoses 21 of 24 cases (87.50%), with all three failures concentrated along the Early Blight vs. Septoria look-alike boundary. Random acquisition (B1) yields zero accuracy gain despite expending 1.0 additional steps, illustrating that undirected evidence acquisition injects variance without resolving ambiguity. B4 degrades accuracy to 83.33% by querying weather context when morphological lesion inspection was required.

Both B5 and EvidenceGain-X (B6) achieve peak diagnostic accuracy of **91.67%** and a **Macro F1 of 0.9188**, resolving ambiguous look-alikes. EvidenceGain-X achieves this in an average of **0.88 steps** (a 12% inquiry reduction compared to static acquisition B2), driven by conservative stopping on clean cases.

### B. Entropy Reduction and Information Gain
EvidenceGain-X achieves an average entropy reduction of **0.359 bits** on ambiguous cases—a **62.4% greater information gain** than random querying (0.221 bits) and 26.4% higher than fixed sequencing (0.284 bits). This confirms that counterfactual simulation accurately identifies the single most discriminative physical clue for each specific confusion pair.

### C. Adversarial Stress-Testing and Safety Gating
To evaluate cyber-physical resilience, 50 degraded and contradictory inputs were injected into the test pipeline:
- **Defocus & Motion Blur ($\sigma^2 < 80.0$):** B0 assigned an average 89.5% confidence of Late Blight to meaningless blur. Module 7 intercepted **100% of blurry uploads**, completely preventing Softmax hallucination.
- **Specimen Mismatch & Healthy Contradiction:** When a healthy leaf was uploaded into an active Early Blight episode (where Healthy prior was 0.3%), Module 8 detected specimen incongruence ($S_{\text{contra}} = 0.982$, prior $< 0.05$), quarantining the input and preserving prior diagnostic belief at 71.4%. Baselines B0–B5 incorporated the foreign input, destabilizing the diagnosis.

### D. Component Ablation Analysis
Table II quantifies the operational necessity of each component.

```
TABLE II
SYSTEM COMPONENT ABLATION STUDY

Ablation Configuration                  Test Accuracy   Mean Steps   Adversarial Pass Rate
-------------------------------------------------------------------------------------------
Full EvidenceGain-X (Proposed)             91.67%          0.88             100.0%
w/o Hypothesis Separation (alpha = 0)      87.50%          0.88             100.0%
w/o Diagnostic Power (gamma = 0)           87.50%          0.92             100.0%
w/o Uncertainty Reduction (beta = 0)       91.67%          1.12             100.0%
w/o Cost Penalty (lambda = 0)              91.67%          1.25             100.0%
w/o Optical Blur Gating (Module 7)         58.33%          0.88               0.0%
w/o Contradiction Guard (Module 8)         75.00%          0.88               0.0%
-------------------------------------------------------------------------------------------
```
Removing Module 7 causes catastrophic accuracy collapse to 58.33% under field noise, proving that input quality gating is an indispensable prerequisite for trustworthy deployment. Removing cost penalization ($\lambda = 0$) increases farmer inquiry burden by 42% (1.25 steps) without diagnostic gain.

### E. Latency and Edge Footprint
Benchmarked on an Intel Core i7-12700K (single-thread CPU mode):
- ResNet-18 forward pass: **12.1 ms**
- Platt scaling & entropy: **0.02 ms**
- Module 7 Laplacian variance: **1.8 ms**
- Module 5 Counterfactual simulation: **0.8 ms**
- Module 8 Contradiction check: **0.4 ms**
- **Total Diagnostic Loop Latency:** **< 30 ms** on CPU (1.45 s full Streamlit UI render).
Memory footprint totals **44.5 MB**, permitting offline deployment on sub-$100 agricultural edge microcomputers.

---

## VII. LIMITATIONS AND ETHICAL IMPLICATIONS
While EvidenceGain-X addresses key vulnerabilities of single-shot models, several limitations remain. First, the 24-episode benchmark, while rigorously plant-grouped, is limited in statistical scale due to the physical scarcity of complete multi-modal foliar episodes. Second, expanding to wider crop taxonomies (e.g., cassava, rice, maize) requires recalibrating the discriminative affinity matrix $\mathbf{A}$ through agronomic literature or automated knowledge-graph extraction. Ethically, EvidenceGain-X is designed as an advisory clinical decision-support tool; high-consequence interventions (such as chemical spraying or quarantine culling) should remain subject to human verification.

---

## VIII. CONCLUSION AND FUTURE WORK
This paper introduced **EvidenceGain-X**, an autonomous active diagnostic framework that resolves the core failure modes of single-shot agricultural AI. By combining temperature-scaled uncertainty quantification, counterfactual evidence planning, multi-objective utility optimization, and dual cyber-physical safety shields (optical blur gating and specimen mismatch detection), EvidenceGain-X elevates tomato pathology accuracy from 87.50% to 91.67% in an average of 0.88 sequential observation steps while achieving 100% rejection against corrupted and contradictory inputs.

Future research will focus on INT8 TensorRT quantization for ultra-low-power microcontrollers, expanding the multimodal taxonomy to multi-crop disease complexes, and integrating conversational audio interfaces in regional languages to maximize accessibility for smallholder farming communities worldwide.

---

## REFERENCES
[1] D. P. Hughes and M. Salathé, "An open access repository of images on plant health to enable the development of mobile disease diagnostics," *arXiv preprint arXiv:1511.08060*, 2015.  
[2] S. P. Mohanty, D. P. Hughes, and M. Salathé, "Using deep learning for image-based plant disease detection," *Frontiers in Plant Science*, vol. 7, p. 1419, Sep. 2016.  
[3] G. N. Agrios, *Plant Pathology*, 5th ed. Burlington, MA, USA: Elsevier Academic Press, 2005.  
[4] K. P. Ferentinos, "Deep learning models for plant disease detection and diagnosis," *Computers and Electronics in Agriculture*, vol. 145, pp. 311–318, Feb. 2018.  
[5] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Las Vegas, NV, USA, 2016, pp. 770–778.  
[6] A. Ramcharan, K. Baranowski, P. McCloskey, B. Ahmed, J. Legg, and D. P. Hughes, "Deep learning for image-based cassava disease detection," *Frontiers in Plant Science*, vol. 8, p. 1852, Oct. 2017.  
[7] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. 34th Int. Conf. Mach. Learn. (ICML)*, Sydney, NSW, Australia, 2017, pp. 1321–1330.  
[8] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian approximation: Representing model uncertainty in deep learning," in *Proc. 33rd Int. Conf. Mach. Learn. (ICML)*, New York, NY, USA, 2016, pp. 1050–1059.  
[9] B. Lakshminarayanan, A. Pritzel, and C. Blundell, "Simple and scalable predictive uncertainty estimation using deep ensembles," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, Long Beach, CA, USA, 2017, pp. 6402–6413.  
[10] B. Settles, "Active learning literature survey," Comput. Sci. Dept., Univ. Wisconsin-Madison, Tech. Rep. 1648, 2009.  
[11] K. Chaloner and I. Verdinelli, "Bayesian experimental design: A review," *Stat. Sci.*, vol. 10, no. 3, pp. 273–304, Aug. 1995.  
[12] D. V. Lindley, "On a measure of the information provided by an experiment," *Ann. Math. Stat.*, vol. 27, no. 4, pp. 986–1005, Dec. 1956.  
[13] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei, "ImageNet: A large-scale hierarchical image database," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, Miami, FL, USA, 2009, pp. 248–255.  
[14] J. Pech-May, F. J. Sanchez-Marin, and J. A. Ramirez-Cortes, "Laplacian-based focus measure operators for shape from focus: A comparative study," *Opt. Lasers Eng.*, vol. 128, p. 106020, May 2020.  
[15] J. Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed. Cambridge, UK: Cambridge Univ. Press, 2009.  
