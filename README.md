# SURAKSHA-DRISHTI (सुरक्षा दृष्टि)
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
* **Role-Based Information Masking:** Company Commanders only view aggregated unit strain distributions and operational rosters; confidential personal reflections and evidence chains are accessible solely by authorized Medical / Unit Welfare Officers.
* **Air-Gapped Local Edge Runtime:** Engineered to operate completely offline on edge hardware without reliance on external commercial cloud infrastructures.

---

## 2. Multi-Role System Hierarchy
SURAKSHA-DRISHTI GATEWAY
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        ▼                          ▼                          ▼
[Personnel Portal]        [Commander Portal]       [Welfare Officer Portal]
• Authorised HRMS Duty    • Unit Workload Grid     • Dynamic Case Triage
• Standardized PHQ-4      • Fatigue Distribution   • Dynamic SHAP Local XAI
• Contactless rPPG Pulse  • AI Rebalancing Advisor • 30-Day Longitudinal Trend
• Voice Pitch Variation   • Human-in-the-Loop Sign • Confidential Referral Dispatch
• Adaptive AI Dialogue
• 4-4-4-4 Box Breathing
---

## 3. Algorithmic & Telemetry Methodology

### Multi-Modal Feature Ingestion
1. **Administrative HRMS Strain Markers:** Days elapsed since sanctioned leave, consecutive night-watch shifts (7-day window), operational hardship zone (Base vs. CI vs. High Altitude), and critical incident exposure.
2. **Psychiatric Self-Screening (PHQ-4):** Integrates the ultra-brief, clinically validated GAD-2 (anxiety) and PHQ-2 (depression) standardized screeners.
3. **Optional Facial rPPG Telemetry:** Contactless optical pulse extraction via Green-channel chrominance analysis ($520\text{–}570\text{ nm}$ absorption peak) filtered through a 3rd-order Butterworth bandpass filter ($0.75\text{–}3.0\text{ Hz}$).
4. **Voice Pitch Variability Proxy:** Frame-by-frame acoustic pitch period autocorrelation measuring vocal tract perturbation as a contextual physiological strain indicator.
5. **Predictive Engine:** Local Random Forest Classifier computing risk indices ($0\text{ to }100$) mapped into actionable welfare bands.
6. **Explainable AI (XAI):** `shap.TreeExplainer` computing mathematical Shapley feature attributions dynamically per test profile.

---

## 4. Scientific Methodology & Dataset Scope

* **Prototype Dataset Notice:** The machine learning model is trained on a synthetic longitudinal personnel dataset designed to demonstrate pipeline feasibility, multi-modal feature fusion, and SHAP interpretability.
* **Operational Calibration Notice:** The reported ROC-AUC demonstrates technical convergence on the synthesized cohort. Real-world force deployment requires authorized institutional records for final operational calibration and clinical validation.
* **Telemetry Context:** Optical pulse and acoustic pitch variations represent auxiliary autonomic signals and are never interpreted in isolation as standalone psychological determinants.

---

## 5. Local Setup & Execution Guide

### Prerequisites
* Python 3.10 or 3.11
* Integrated/USB Webcam & Microphone (for optional telemetry demo)

### Installation
```bash
# 1. Clone private repository
git clone https://github.com/ankanmahanti3-prog/suraksha-drishti.git
cd suraksha-drishti

# 2. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# 3. Install clean dependencies
pip install -r requirements.txt

# 4. Launch Air-Gapped Platform
streamlit run app.py