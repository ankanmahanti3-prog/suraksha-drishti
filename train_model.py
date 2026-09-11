"""
SURAKSHA-DRISHTI: Machine Learning Training Pipeline
Aligned with MHA CRPF Problem Statement ID: 26186

Note on Dataset Scope:
This script synthesizes independent cross-sectional personnel records
(2,500 samples) across administrative, self-reported, and auxiliary
physiological indicators to demonstrate pipeline feasibility, feature
fusion, and local SHAP explainability. It does not represent repeated
longitudinal observations of the same individuals over time.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# Set deterministic random seed for pipeline reproducibility
np.random.seed(42)
N_SAMPLES = 2500

# 1. Synthesize Independent Personnel Duty & Operational Attributes
days_since_leave = np.random.exponential(scale=35, size=N_SAMPLES).clip(0, 300)
night_shifts_7d = np.random.poisson(lam=2.0, size=N_SAMPLES).clip(0, 7)
deployment_hardship = np.random.choice([0, 1, 2], p=[0.5, 0.35, 0.15], size=N_SAMPLES)
trauma_incident_flag = np.random.choice([0, 1], p=[0.85, 0.15], size=N_SAMPLES)

# 2. Synthesize Self-Reported Wellness & Recovery Indicators
sleep_hours = np.random.normal(loc=6.2, scale=1.4, size=N_SAMPLES).clip(2.0, 10.0)
subjective_fatigue = np.random.choice([1, 2, 3, 4], p=[0.4, 0.35, 0.18, 0.07], size=N_SAMPLES)

# 3. Synthesize Optional Auxiliary Physiological & Acoustic Signals
optional_vocal_jitter = np.random.normal(loc=1.2, scale=0.6, size=N_SAMPLES).clip(0.3, 5.5)
optional_bpm_delta = np.random.normal(loc=4.0, scale=10.0, size=N_SAMPLES).clip(-15.0, 45.0)

# 4. Construct Operational Latent Risk Target (Synthetic Ground Truth Formulation)
latent_risk = (
    0.28 * (days_since_leave / 90.0) +
    0.22 * (night_shifts_7d / 5.0) +
    0.15 * (deployment_hardship / 2.0) +
    0.20 * trauma_incident_flag +
    0.22 * ((8.0 - sleep_hours) / 4.0) +
    0.18 * ((subjective_fatigue - 1) / 3.0) +
    0.08 * (optional_vocal_jitter / 3.0) +
    0.07 * (optional_bpm_delta / 20.0) +
    np.random.normal(0, 0.12, N_SAMPLES)
)

# Binarize into discrete strain bands (Threshold: 0.52)
y = (latent_risk > 0.52).astype(int)

# 5. Assemble Structured Training Dataframe
X = pd.DataFrame({
    'Days_Since_Leave': days_since_leave,
    'Night_Shifts_7d': night_shifts_7d,
    'Deployment_Hardship': deployment_hardship,
    'Trauma_Incident_Flag': trauma_incident_flag,
    'Sleep_Hours': sleep_hours,
    'Subjective_Fatigue': subjective_fatigue,
    'Optional_Vocal_Jitter': optional_vocal_jitter,
    'Optional_BPM_Delta': optional_bpm_delta
})

# 6. Train-Validation Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 7. Model Instantiation & Training
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    min_samples_split=8,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# 8. Feasibility Evaluation Output
y_prob = clf.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"Prototype Model Feasibility ROC-AUC (Synthetic Validation Split): {auc:.4f}")
print("\nClassification Feasibility Summary:\n", classification_report(y_test, clf.predict(X_test)))

# 9. Serialize Model Artifacts
joblib.dump(clf, 'welfare_rf_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print("Model artifacts successfully serialized: welfare_rf_model.pkl, feature_names.pkl")