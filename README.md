# SURAKSHA-DRISHTI
### AI-Assisted Personnel Welfare & Operational Stress-Risk System
**Aligned with Ministry of Home Affairs (MHA) & CRPF Problem Statement ID: 26186**

---

## 1. System Architecture & Core Philosophy
SURAKSHA-DRISHTI is an early-warning, multi-tier decision-support platform designed to predict and mitigate cumulative operational exhaustion, burnout, and stress-risk among armed forces personnel.

> **Ethical Design Axiom:**  
> *"AI identifies welfare-risk patterns; authorised humans decide interventions."*  
> The system provides predictive operational risk indicators—**never clinical or psychiatric diagnoses**.

### Key Architectural Firewalls
* **Non-Disciplinary Data Firewall:** All voluntary screenings, acoustic telemetry, and self-assessment dialogues are strictly firewalled from Annual Confidential Reports (ACRs), seniority evaluations, and promotional boards.
* **Role-Based Portal Isolation (Prototype):** Company Commanders only view aggregated unit strain distributions and operational rosters; confidential personal reflections and evidence chains are accessible solely by authorized Medical / Unit Welfare Officers.
* **Air-Gapped Local Edge Runtime:** Engineered to operate completely offline on edge hardware without external web font imports or commercial cloud infrastructures.

---

## 2. Multi-Role System Hierarchy

```
                            SURAKSHA-DRISHTI GATEWAY
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
    [Personnel Portal]        [Commander Portal]       [Welfare Officer Portal]
    • Authorised HRMS Duty    • Unit Workload Grid     • Dynamic Case Triage
    • Mental Wellbeing PHQ-4  • Fatigue Distribution   • Dynamic SHAP Local XAI
    • Contactless rPPG Pulse  • AI Rebalancing Advisor • In-Session Trajectory Tracking
    • Voice Pitch Variation   • Human-in-the-Loop Sign • Simulated Workflow Dispatch
    • Guided Welfare Dialogue
    • 4-4-4-4 Box Breathing
```

---

## 3. Algorithmic & Telemetry Methodology

### Multi-Modal Feature Ingestion
1. **Administrative Duty Profile (Mock HRMS):** Days elapsed since sanctioned leave, consecutive night-watch shifts (7-day window), operational hardship zone (Base vs. CI vs. High Altitude), and critical incident exposure.
2. **Mental Wellbeing Self-Screening (PHQ-4):** Integrates the ultra-brief, clinically validated GAD-2 (anxiety) and PHQ-2 (depression) standardized screeners.
3. **Optional Facial rPPG Telemetry with Quality Gate:** Contactless optical pulse extraction via Green-channel chrominance analysis (520–570 nm absorption peak) filtered through a 3rd-order Butterworth bandpass filter (0.75–3.0 Hz) with variance and signal-to-noise ratio verification.
4. **Voice Pitch Variability Proxy:** Frame-by-frame acoustic pitch period autocorrelation measuring vocal tract perturbation as a contextual physiological strain indicator.
5. **Predictive Engine:** Local Random Forest Classifier computing risk indices (0 to 100) mapped into actionable welfare bands.
6. **Explainable AI (XAI):** `shap.TreeExplainer` computing mathematical Shapley feature attributions dynamically per test profile.

---

## 4. Scientific Methodology & Dataset Scope

* **Prototype Dataset Scope:** The machine learning model is trained on a **synthetic personnel-risk dataset** (2,500 synthesized independent duty profiles) engineered to validate multi-modal feature fusion, inference pipelines, and SHAP explainability. It does not consist of real historical clinical records.
* **ROC-AUC Metric Notice:** The model's validation performance score of **0.879 ROC-AUC** reflects mathematical convergence on the synthetic prototype dataset. It must **not** be interpreted as clinical accuracy, real-world accuracy, or operational military prediction accuracy. Real-world deployment requires authorized longitudinal institutional records for domain calibration.
* **Telemetry Context:** Optical pulse estimates and acoustic pitch variations are non-diagnostic, contextual proxies and are never evaluated as standalone psychiatric determinants.
* **Simulated Workflow Actions:** Intervention routing buttons (leave reviews, buddy systems, counseling requests) represent simulated prototype workflows for demonstration purposes.

---

## 5. Local Setup & Execution Guide

### Prerequisites
* Python 3.10 or 3.11
* Integrated/USB Webcam & Microphone (for optional telemetry demo)

### Installation
```bash
# 1. Clone private repository
git clone [https://github.com/ankanmahanti3-prog/suraksha-drishti.git](https://github.com/ankanmahanti3-prog/suraksha-drishti.git)
cd suraksha-drishti

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install pinned dependencies (scikit-learn is pinned to 1.9.0 to match the serialized model artifact)
pip install -r requirements.txt

# 4. Launch Air-Gapped Platform
streamlit run app.py
```

### Demonstration Accounts (Role-Based Access)

| Role | Service ID | PIN | Scope / Permissions |
| :--- | :--- | :--- | :--- |
| **Personnel (Elevated Check-in)** | `P-10001` | `1234` | Full check-in, live telemetry scan, guided dialogue, self-help tools. |
| **Personnel (Balanced Baseline)** | `P-10002` | `1234` | Nominal duty load view, check-in, guided breathing. |
| **Company Commander** | `C-20001` | `2345` | High-level fatigue grid, prototype roster rebalancing advisor. |
| **Unit Welfare Officer** | `W-30001` | `3456` | Live triage, evidence chains, SHAP XAI charts, simulated intervention routing. |
| **Security Negative Test** | `P-99999` | `*` | **Access Denied** — proves RBAC authentication check. |