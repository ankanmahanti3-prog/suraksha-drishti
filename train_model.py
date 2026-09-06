import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1. Generate Synthetic Longitudinal CAPF Dataset (N=2,500 records)
np.random.seed(42)
n_samples = 2500

days_since_leave = np.random.randint(5, 140, n_samples)
consecutive_night_shifts = np.random.randint(0, 8, n_samples)
deployment_hardship = np.random.choice([0, 1, 2], p=[0.4, 0.35, 0.25], size=n_samples) # 0=Peace, 1=CI, 2=High Altitude
trauma_incident_flag = np.random.binomial(1, 0.15, n_samples)
sleep_hours = np.random.normal(5.8, 1.4, n_samples).clip(2.0, 9.0)
subjective_fatigue = np.random.randint(1, 5, n_samples) # 1=Low, 4=Severe
opt_vocal_jitter = np.random.normal(1.8, 1.1, n_samples).clip(0.4, 6.5)
opt_bpm_delta = np.random.normal(6.0, 9.0, n_samples).clip(-10, 35)

# Ground Truth Risk Probability Logic (Simulating Longitudinal Burnout/Stress Risk Trend)
latent_risk = (
    0.30 * (days_since_leave / 120.0) +
    0.25 * (consecutive_night_shifts / 7.0) +
    0.15 * (deployment_hardship / 2.0) +
    0.20 * trauma_incident_flag +
    0.25 * ((8.0 - sleep_hours) / 6.0) +
    0.15 * (subjective_fatigue / 4.0) +
    0.10 * (opt_vocal_jitter / 5.0) +
    0.05 * (opt_bpm_delta / 30.0) +
    np.random.normal(0, 0.08, n_samples)
)

# Binary Target: 1 = Elevated Welfare Risk Trend, 0 = Nominal Trend
y = (latent_risk > 0.52).astype(int)

X = pd.DataFrame({
    'Days_Since_Leave': days_since_leave,
    'Night_Shifts_7d': consecutive_night_shifts,
    'Deployment_Hardship': deployment_hardship,
    'Trauma_Incident_Flag': trauma_incident_flag,
    'Sleep_Hours': sleep_hours,
    'Subjective_Fatigue': subjective_fatigue,
    'Optional_Vocal_Jitter': opt_vocal_jitter,
    'Optional_BPM_Delta': opt_bpm_delta
})

# 2. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Random Forest Classifier (Non-linear Risk Modeling)
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 4. Validation Metrics (Proof for judges)
y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_pred_proba)
print(f">> Model Training Complete.")
print(f">> ROC-AUC Score: {auc:.3f}")
print(">> Global Feature Importances:")
for col, imp in zip(X.columns, model.feature_importances_):
    print(f"   - {col:25s}: {imp*100:.1f}%")

# 5. Export Model & Artifacts
joblib.dump(model, 'welfare_rf_model.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
print(">> Artifacts saved: welfare_rf_model.pkl, feature_names.pkl")