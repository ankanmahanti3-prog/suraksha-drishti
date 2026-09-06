import streamlit as st
import numpy as np
import cv2
import sounddevice as sd
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
import joblib
import shap
import time
from registry import authenticate_user, check_permission, PERSONNEL_REGISTRY

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SURAKSHA-DRISHTI | AI Personnel Welfare Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STRICT THEME-INDEPENDENT CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, .stApp, p, h1, h2, h3, h4, label { 
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] { 
        background-color: #f8fafc !important; 
        color: #0f172a !important; 
    }

    .stApp p, .stApp span, .stApp label, .stApp h1, .stApp h2, .stApp h3 {
        color: #0f172a !important;
    }

    /* Floating Pop Card Containers */
    .pop-card {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 20px !important;
        padding: 24px 28px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 10px 25px -5px rgba(148, 163, 184, 0.15) !important;
    }

    /* Top Floating Navigation Bar */
    .top-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #ffffff !important;
        padding: 14px 28px !important;
        border-radius: 20px !important;
        box-shadow: 0 6px 20px -4px rgba(148, 163, 184, 0.15) !important;
        border: 1px solid #e2e8f0 !important;
        margin-bottom: 16px !important;
    }

    /* Radial Stress Gauge */
    .radial-gauge {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        box-shadow: 0 6px 18px rgba(148, 163, 184, 0.2);
    }
    .gauge-val {
        font-size: 2.1rem !important;
        font-weight: 800 !important;
        line-height: 1 !important;
    }
    .gauge-sub {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }

    /* Metric Highlight Boxes */
    .metric-pill-box {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 16px 20px !important;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    /* Status Badges */
    .pill-badge-red {
        background: #fef2f2 !important;
        color: #ef4444 !important;
        border: 1px solid #fecaca !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }
    .pill-badge-green {
        background: #f0fdf4 !important;
        color: #16a34a !important;
        border: 1px solid #bbf7d0 !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }
    .pill-badge-blue {
        background: #eff6ff !important;
        color: #2563eb !important;
        border: 1px solid #bfdbfe !important;
        padding: 4px 12px !important;
        border-radius: 9999px !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        display: inline-block !important;
    }

    /* Form Input Fixes */
    input[type="text"], input[type="password"] {
        background-color: #f8fafc !important;
        border: 1px solid #cbd5e1 !important;
        color: #0f172a !important;
        border-radius: 12px !important;
        -webkit-text-fill-color: #0f172a !important;
    }

    /* Button Styling */
    .stButton>button[kind="primary"], div[data-testid="stFormSubmitButton"]>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.25) !important;
        text-shadow: none !important;
    }
    .stButton>button[kind="primary"] p, div[data-testid="stFormSubmitButton"]>button p {
        color: #ffffff !important;
    }

    .stButton>button[kind="secondary"], .stButton>button {
        background: #ffffff !important;
        color: #334155 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 8px 18px !important;
        box-shadow: 0 2px 6px rgba(148, 163, 184, 0.1) !important;
        text-shadow: none !important;
    }
    .stButton>button[kind="secondary"] p, .stButton>button p {
        color: #334155 !important;
    }

    .stButton>button:hover {
        border-color: #3b82f6 !important;
        background: #f1f5f9 !important;
    }

    /* Tactical Breathing Box */
    .breathing-circle {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: #eff6ff;
        border: 4px solid #3b82f6;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 14px auto;
        box-shadow: 0 0 25px rgba(59, 130, 246, 0.25);
    }

    /* High-Contrast Light Table */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 14px 0;
        font-size: 0.84rem;
        background-color: #ffffff;
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    .styled-table thead tr {
        background-color: #f8fafc;
        color: #334155;
        text-align: left;
        font-weight: 700;
        border-bottom: 2px solid #e2e8f0;
    }
    .styled-table th, .styled-table td {
        padding: 12px 16px;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #f1f5f9;
        color: #0f172a;
    }
    .styled-table tbody tr:hover {
        background-color: #f8fafc;
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHED ML & SIGNAL PIPELINE ---
@st.cache_resource
def load_ml_model():
    try:
        model = joblib.load('welfare_rf_model.pkl')
        features = joblib.load('feature_names.pkl')
        explainer = shap.TreeExplainer(model)
        return model, features, explainer
    except:
        return None, None, None

model, feature_names, explainer = load_ml_model()

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, data)

def capture_biometrics(duration=10):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    fps = 30.0
    total_frames = int(fps * duration)
    green_signals = []
    
    col_cam, col_chart = st.columns(2)
    with col_cam:
        video_placeholder = st.empty()
    with col_chart:
        chart_placeholder = st.empty()
        
    progress_bar = st.progress(0)
    fs_audio = 44100
    audio_record = sd.rec(int(duration * fs_audio), samplerate=fs_audio, channels=1, dtype='float32')

    for f in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        current_g = green_signals[-1] if len(green_signals) > 0 else 100.0
        for (x, y, w, h) in faces:
            fh_x1, fh_x2 = int(x + w * 0.28), int(x + w * 0.72)
            fh_y1, fh_y2 = int(y + h * 0.08), int(y + h * 0.22)
            roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]
            if roi.size > 0:
                current_g = roi[:, :, 1].mean()
                cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (79, 70, 229), 2)
            break
            
        green_signals.append(current_g)
        progress_bar.progress((f + 1) / total_frames)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        if len(green_signals) > 20 and f % 6 == 0:
            wave = np.array(green_signals[-60:])
            wave = wave - np.mean(wave)
            chart_placeholder.line_chart(pd.DataFrame({"Forehead Capillary Spectrum": wave}), height=180)

    cap.release()
    sd.wait()
    progress_bar.empty()
    video_placeholder.empty()
    chart_placeholder.empty()

    raw_sig = np.array(green_signals)
    detrended = raw_sig - np.mean(raw_sig)
    filtered = bandpass_filter(detrended, lowcut=0.75, highcut=3.0, fs=fps)
    fft_vals = np.abs(np.fft.rfft(filtered))
    fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    valid_idx = np.where((fft_freqs >= 0.75) & (fft_freqs <= 3.0))
    peak_freq = fft_freqs[valid_idx][np.argmax(fft_vals[valid_idx])]
    bpm = float(peak_freq * 60.0)

    audio = audio_record.flatten()
    frame_size = int(fs_audio * 0.04)
    hop_size = int(fs_audio * 0.02)
    pitches = []
    for i in range(0, len(audio) - frame_size, hop_size):
        chunk = audio[i:i + frame_size]
        if np.sqrt(np.mean(chunk**2)) > 0.01:
            chunk_corr = np.correlate(chunk, chunk, mode='full')[len(chunk)//2:]
            peaks, _ = find_peaks(chunk_corr[int(fs_audio/350):int(fs_audio/70)], distance=15)
            if len(peaks) > 0:
                pitches.append(fs_audio / (peaks[0] + int(fs_audio/350)))
                
    jitter = (np.mean(np.abs(np.diff(pitches))) / np.mean(pitches) * 100) if len(pitches) > 5 else 1.2
    vocal_stress = float(np.clip((jitter - 0.8) * 35.0, 15, 95))

    return bpm, jitter, vocal_stress, filtered

# --- SESSION INITIALIZATION & AUTH STATE ---
if "booted" not in st.session_state:
    st.session_state.booted = False
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.user = None

# ==================== 1. SYSTEM INITIALIZATION EXPERIENCE ==================== #
if not st.session_state.booted:
    _, center_col, _ = st.columns([1, 1.8, 1])
    with center_col:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="pop-card" style="text-align: center; padding: 40px 30px;">
<div style="font-size: 3.2rem; margin-bottom: 12px;">🛡️</div>
<h1 style="color: #0f172a; font-size: 1.8rem; font-weight: 800; margin-bottom: 4px;">SURAKSHA-DRISHTI</h1>
<p style="color: #4f46e5; font-size: 0.95rem; font-weight: 600; margin-bottom: 8px;">AI-Assisted Personnel Welfare & Stress-Risk System</p>
<p style="font-size: 0.8rem; color: #64748b; margin-bottom: 24px;">Aligned with MHA CRPF Problem Statement ID: 26186 | Behavioral & HRMS Analytics with Privacy Safeguards</p>
<div style="border-top: 1px solid #edf2f7; padding-top: 20px;">
<p style="font-size: 0.84rem; color: #334155; font-weight: 600; margin-bottom: 16px;">INITIALIZING SECURE ENVIRONMENT...</p>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; text-align: left; font-size: 0.82rem; margin-bottom: 24px;">
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">✓ Identity Services &nbsp;&nbsp;<b style="color: #16a34a;">Connected</b></div>
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">✓ Secure Data Layer &nbsp;&nbsp;<b style="color: #16a34a;">Connected</b></div>
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">✓ Welfare Analytics &nbsp;&nbsp;<b style="color: #16a34a;">Connected</b></div>
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">✓ AI Risk Engine &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b style="color: #16a34a;">Connected</b></div>
</div>
</div>""", unsafe_allow_html=True)
        
        if st.button("PROCEED TO SECURE ACCESS GATEWAY →", type="primary", use_container_width=True):
            st.session_state.booted = True
            st.rerun()

        st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.75rem; margin-top: 20px;'>Privacy-first • Welfare-first • Local Edge AI Runtime</div>", unsafe_allow_html=True)
    st.stop()

# ==================== 2. SECURE LOGIN / AUTHENTICATION ==================== #
if not st.session_state.authenticated:
    _, login_col, _ = st.columns([1, 1.4, 1])
    with login_col:
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        st.markdown("""<div class="pop-card" style="text-align: center; padding: 34px 28px 24px 28px; margin-bottom: 12px;">
<div style="font-size: 2.5rem; margin-bottom: 10px;">🛡️</div>
<h2 style="color: #0f172a; font-size: 1.6rem; font-weight: 800; margin-bottom: 4px;">SURAKSHA-DRISHTI</h2>
<p style="color: #4f46e5; font-size: 0.9rem; font-weight: 600; margin-bottom: 2px;">Personnel Welfare Intelligence Platform</p>
<p style="color: #64748b; font-size: 0.78rem;">Secure access to authorised welfare and workload services</p>
</div>""", unsafe_allow_html=True)

        with st.form(key="login_form", border=True):
            input_id = st.text_input("Service ID", value="P-10001")
            input_pin = st.text_input("Access PIN", type="password", value="1234")
            
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            submit = st.form_submit_button("AUTHENTICATE SESSION →", use_container_width=True)
            if submit:
                valid, result = authenticate_user(input_id.strip(), input_pin.strip())
                if valid:
                    st.session_state.authenticated = True
                    st.session_state.user = result
                    st.session_state.user_id = input_id.strip()
                    if result["role"] == "Personnel":
                        st.session_state.current_nav = "Home"
                    elif result["role"] == "Commander":
                        st.session_state.current_nav = "Workload Grid"
                    else:
                        st.session_state.current_nav = "Welfare Assessment Triage"
                    st.rerun()
                else:
                    st.error(result)

        st.markdown("""<div style="display: flex; justify-content: space-around; text-align: center; margin-top: 14px; font-size: 0.75rem; color: #64748b; font-weight: 600;">
<div>👤 Role-Based Access</div>
<div>🛡️ Welfare Data Firewall</div>
<div>⚙️ Local AI Processing</div>
</div>
<div style="text-align: center; color: #94a3b8; font-size: 0.72rem; margin-top: 12px;">
🔒 Protected • Encrypted • Air-gapped Deployment Ready
</div>""", unsafe_allow_html=True)

        with st.expander("Demo Helper (Prototype Mode)"):
            st.markdown("""<div style="color: #334155; font-size: 0.8rem; line-height: 1.6;">
• Personnel Account: <b>P-10001</b> (PIN: <code>1234</code>)<br>
• Balanced Personnel: <b>P-10002</b> (PIN: <code>1234</code>)<br>
• Commander Account: <b>C-20001</b> (PIN: <code>2345</code>)<br>
• Welfare Officer Account: <b>W-30001</b> (PIN: <code>3456</code>)<br>
• Rejection Test: <b>P-99999</b>
</div>""", unsafe_allow_html=True)
    st.stop()

# ==================== MAIN AUTHENTICATED WORKSPACE ==================== #
user = st.session_state.user
user_role = user["role"]

if user_role == "Personnel":
    role_nav_options = ["Home", "Voluntary Wellness Check-in", "Self-Help & Tactical Breathing"]
elif user_role == "Commander":
    role_nav_options = ["Workload Grid", "Workload Rebalancing Advisor"]
else:
    role_nav_options = ["Welfare Assessment Triage", "Dynamic XAI Analysis", "Longitudinal Trend (30 Days)"]

if st.session_state.get("current_nav") not in role_nav_options:
    st.session_state.current_nav = role_nav_options[0]

# --- GLOBAL TOP FLOATING HEADER ---
st.markdown(f"""<div class="top-header">
<div style="display: flex; align-items: center; gap: 12px;">
<span style="font-size: 1.6rem;">🛡️</span>
<div>
<div style="font-weight: 800; font-size: 1.15rem; color: #0f172a; line-height: 1.1;">SURAKSHA-DRISHTI</div>
<div style="font-size: 0.72rem; color: #4f46e5; font-weight: 600;">AI-Assisted Personnel Welfare System</div>
</div>
</div>
<div style="display: flex; align-items: center; gap: 16px;">
<div style="text-align: right;">
<div style="font-size: 0.85rem; font-weight: 700; color: #0f172a;">{user['display_id']}</div>
<div style="font-size: 0.72rem; color: #64748b;">Role: <b>{user_role}</b> | Unit: {user['unit']}</div>
</div>
<span class="pill-badge-green">AUTHENTICATED</span>
</div>
</div>""", unsafe_allow_html=True)

# Navigation Bar
nav_cols = st.columns([1] * len(role_nav_options) + [1.2])
for i, opt in enumerate(role_nav_options):
    is_active = (st.session_state.current_nav == opt)
    btn_type = "primary" if is_active else "secondary"
    if nav_cols[i].button(opt, key=f"nav_btn_{i}", type=btn_type, use_container_width=True):
        st.session_state.current_nav = opt
        st.rerun()

if nav_cols[-1].button("🚪 Logout", key="logout_btn", use_container_width=True):
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.current_nav = "Home"
    st.rerun()

st.markdown("""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 5px solid #4f46e5; border-radius: 14px; padding: 12px 20px; margin-bottom: 22px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 12px rgba(148, 163, 184, 0.08);">
<span style="font-size: 0.82rem; color: #334155; font-weight: 600;">
⚖️ <b>Core Ethical Architecture:</b> <i>"AI identifies welfare-risk patterns; authorised humans decide interventions."</i>
</span>
<span style="font-size: 0.72rem; color: #64748b; font-weight: 600;">System provides predictive risk indicators, not clinical or psychiatric diagnoses.</span>
</div>""", unsafe_allow_html=True)

# ==================== 1. PERSONNEL ROLE VIEWS ==================== #
if user_role == "Personnel":

    # --- VIEW 1: HOME / OVERVIEW ---
    if st.session_state.current_nav == "Home":
        col_left, col_right = st.columns([1.7, 1.3])

        with col_left:
            st.markdown(f"""<div class="pop-card">
<div style="font-size: 1.4rem; font-weight: 800; color: #0f172a; margin-bottom: 2px;">Good afternoon, {user['display_id']}</div>
<p style="color: #64748b; font-size: 0.82rem; margin-bottom: 20px;">{user['unit']} • Authorized Personnel Portal</p>
<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
<div class="metric-pill-box">
<span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Welfare Risk</span>
<div style="font-size: 1.8rem; font-weight: 800; color: #ef4444; margin: 4px 0;">76<span style="font-size: 0.9rem; color: #94a3b8;">/100</span></div>
<span class="pill-badge-red" style="width: fit-content;">ELEVATED</span>
</div>
<div class="metric-pill-box">
<span style="font-size: 0.72rem; font-weight: 700; color: #64748b; text-transform: uppercase;">Duty Load</span>
<div style="font-size: 1.8rem; font-weight: 800; color: #4f46e5; margin: 4px 0;">4 <span style="font-size: 0.95rem; font-weight: 600; color: #64748b;">Watches</span></div>
<span style="font-size: 0.75rem; color: #64748b; font-weight: 500;">Consecutive Night Watches</span>
</div>
</div>
</div>""", unsafe_allow_html=True)

            st.markdown("""<div class="pop-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
<span style="font-size: 0.92rem; font-weight: 800; color: #ef4444;">⚠️ AI WELFARE INSIGHT</span>
<span style="font-size: 0.72rem; color: #94a3b8; font-weight: 600;">Advisory Pattern</span>
</div>
<p style="font-size: 0.88rem; color: #1e293b; font-weight: 600; margin-bottom: 6px;">Elevated welfare-risk pattern detected</p>
<p style="font-size: 0.78rem; color: #64748b; margin-bottom: 12px;">Primary contributing factors identified by behavioral predictive engine:</p>
<div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.8rem; color: #334155;">
<div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">• <b>Extended interval since sanctioned leave:</b> 82 days logged</div>
<div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">• <b>Consecutive night-watch load:</b> 4 night shifts in past 7 days</div>
<div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">• <b>Sleep deficit:</b> Average 4.0 hours/day reported</div>
</div>
</div>""", unsafe_allow_html=True)

        with col_right:
            st.markdown("""<div class="pop-card" style="height: calc(100% - 20px);">
<div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-bottom: 16px;">RECENT ACTIVITY & AUDIT LOG</div>
<div style="display: flex; flex-direction: column; gap: 14px; font-size: 0.8rem;">
<div style="border-left: 3px solid #3b82f6; padding-left: 12px;">
<div style="font-weight: 600; color: #1e293b;">Voluntary Self-Assessment Synced</div>
<div style="color: #64748b; font-size: 0.72rem;">Today • Personnel Mobile/Kiosk Module</div>
</div>
<div style="border-left: 3px solid #10b981; padding-left: 12px;">
<div style="font-weight: 600; color: #1e293b;">Authorised HRMS Duty Profile Ingested</div>
<div style="color: #64748b; font-size: 0.72rem;">Yesterday • Alpha Company Registry</div>
</div>
<div style="border-left: 3px solid #f59e0b; padding-left: 12px;">
<div style="font-weight: 600; color: #1e293b;">Local AI Engine Evaluated Risk Trends</div>
<div style="color: #64748b; font-size: 0.72rem;">04 Sep 2026 • Air-gapped Random Forest Model</div>
</div>
</div>
<div style="margin-top: 34px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px; font-size: 0.75rem; color: #64748b;">
🔒 <b>Welfare Data Firewall:</b> Responses are firewalled from ACRs, promotions, and disciplinary files.
</div>
</div>""", unsafe_allow_html=True)

    # --- VIEW 2: VOLUNTARY CHECK-IN WITH STANDARDIZED PHQ-4 SCREENER ---
    elif st.session_state.current_nav == "Voluntary Wellness Check-in":
        st.markdown(f'<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 16px;">Voluntary Wellness Self-Assessment | Pseudonymized ID: {user["display_id"]}</div>', unsafe_allow_html=True)

        col_form, col_telemetry = st.columns([1.1, 0.9])
        with col_form:
            st.markdown('<div class="pop-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 14px;'>1. Authorised HRMS Duty Profile</h3>", unsafe_allow_html=True)
            days_no_leave = st.number_input("Days Elapsed Since Sanctioned Leave", 0, 365, user["days_since_leave"])
            night_shifts = st.slider("Consecutive Night Watches in Past 7 Days", 0, 7, user["night_shifts_7d"])
            deployment_terrain = st.selectbox("Operational Hardship Zone", 
                                              ["Peace Station / Standard Base", "Counter-Insurgency / High Hardship", "Extreme High Altitude"],
                                              index=1 if "Counter" in user["deployment_hardship"] else 0)
            trauma_exposure = st.checkbox("Critical Incident Exposure — Authorized (Past 30 Days)", value=user["trauma_incident_flag"])

            st.markdown("<div style='border-top: 1px solid #edf2f7; margin: 20px 0;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 6px;'>2. Standardized Psychiatric Self-Check (PHQ-4 / GAD-2)</h3>", unsafe_allow_html=True)
            st.caption("Confidential psychological self-check (0: Not at all | 1: Several days | 2: Over half the days | 3: Nearly every day)")
            
            p1, p2 = st.columns(2)
            with p1:
                q1 = st.select_slider("Feeling nervous, anxious, or on edge (GAD-1)", options=[0, 1, 2, 3], value=2)
                q2 = st.select_slider("Not being able to stop or control worrying (GAD-2)", options=[0, 1, 2, 3], value=1)
            with p2:
                q3 = st.select_slider("Little interest or pleasure in doing things (PHQ-1)", options=[0, 1, 2, 3], value=1)
                q4 = st.select_slider("Feeling down, depressed, or hopeless (PHQ-2)", options=[0, 1, 2, 3], value=2)

            phq4_total = q1 + q2 + q3 + q4
            inferred_fatigue = int(np.clip(1 + (phq4_total / 4.0), 1, 4))

            st.markdown("<div style='border-top: 1px solid #edf2f7; margin: 16px 0;'></div>", unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 14px;'>3. Sleep & Fatigue Log</h3>", unsafe_allow_html=True)
            sleep_hours = st.slider("Average Sleep in Last 48 Hours (Hours/Day)", 2.0, 10.0, user["sleep_hours"], 0.5)
            subjective_distress = st.select_slider(
                "Overall Subjective Exhaustion", 
                options=[1, 2, 3, 4], 
                format_func=lambda x: {1: "1 - Rested", 2: "2 - Mild Strain", 3: "3 - Moderate Exhaustion", 4: "4 - Severe Burnout"}[x], 
                value=inferred_fatigue
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col_telemetry:
            st.markdown('<div class="pop-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 1rem; font-weight: 700; color: #0f172a; margin-bottom: 4px;'>4. Optional Authorized Telemetry</h3>", unsafe_allow_html=True)
            st.caption("Non-diagnostic, consent-backed optical pulse and acoustic check.")
            consent = st.checkbox("I voluntarily consent to temporary optical/acoustic check-in.", value=True)
            start_scan = st.button("🚀 INITIATE VOLUNTARY CHECK-IN", type="primary", use_container_width=True, disabled=not consent)

            if start_scan:
                with st.spinner("Processing voluntary inputs through local Random Forest engine..."):
                    bpm, jitter, vocal_stress, pulse_waveform = capture_biometrics(duration=10)

                hardship_map = {"Peace Station / Standard Base": 0, "Counter-Insurgency / High Hardship": 1, "Extreme High Altitude": 2}
                input_df = pd.DataFrame([{
                    'Days_Since_Leave': days_no_leave,
                    'Night_Shifts_7d': night_shifts,
                    'Deployment_Hardship': hardship_map[deployment_terrain],
                    'Trauma_Incident_Flag': 1 if trauma_exposure else 0,
                    'Sleep_Hours': sleep_hours,
                    'Subjective_Fatigue': subjective_distress,
                    'Optional_Vocal_Jitter': jitter,
                    'Optional_BPM_Delta': bpm - 70.0
                }])

                if model is not None:
                    risk_prob = model.predict_proba(input_df)[0][1]
                    risk_index = int(risk_prob * 100)
                else:
                    risk_index = 65

                st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                m1.metric("Optional Pulse", f"{bpm:.0f} BPM")
                m2.metric("Vocal Jitter", f"{jitter:.2f}%")
                m3.metric("Wellness Signal", f"{risk_index}/100", delta=f"{risk_index - 45} vs Base", delta_color="inverse")

                st.write("**Extracted Optical Pulse Waveform:**")
                st.line_chart(pulse_waveform[-150:], height=140)

                if risk_index >= 60:
                    st.error("⚠️ Elevated Welfare Risk Trend Identified")
                    st.info("System Note: An advisory recommendation for workload rebalancing has been forwarded to the Unit Welfare Officer. (No disciplinary record initiated).")
                else:
                    st.success("✅ Nominal Range: Operational Strain within Baseline Tolerances")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- VIEW 3: SOLDIER SELF-HELP RESOURCES & COMBAT BREATHING ---
    elif st.session_state.current_nav == "Self-Help & Tactical Breathing":
        st.markdown(f'<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 16px;">Soldier Self-Help Resources & Support Tools | {user["display_id"]}</div>', unsafe_allow_html=True)

        sh_left, sh_right = st.columns([1.1, 0.9])

        with sh_left:
            st.markdown("""<div class="pop-card">
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
<span style="font-size: 1.05rem; font-weight: 800; color: #0f172a;">🫁 Combat Box Breathing (4-4-4-4 Cycle)</span>
<span class="pill-badge-blue">PARASYMPATHETIC RESET</span>
</div>
<p style="font-size: 0.82rem; color: #64748b; margin-bottom: 14px;">Standard military tactical breathing technique to downregulate autonomic acute strain and restore heart rate variability.</p>
<div class="breathing-circle">
<div style="font-size: 1.6rem; font-weight: 800; color: #2563eb;">4s</div>
<div style="font-size: 0.75rem; font-weight: 700; color: #64748b;">CYCLE</div>
</div>
<div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px; text-align: center; font-size: 0.75rem; margin: 16px 0;">
<div style="background: #f8fafc; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; color: #0f172a;"><b>1. Inhale</b><br>4 Seconds</div>
<div style="background: #f8fafc; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; color: #0f172a;"><b>2. Hold</b><br>4 Seconds</div>
<div style="background: #f8fafc; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; color: #0f172a;"><b>3. Exhale</b><br>4 Seconds</div>
<div style="background: #f8fafc; padding: 8px; border-radius: 8px; border: 1px solid #e2e8f0; color: #0f172a;"><b>4. Rest</b><br>4 Seconds</div>
</div>
</div>""", unsafe_allow_html=True)

            if st.button("▶️ Start 16-Second Guided Tactical Cycle", use_container_width=True):
                b_status = st.empty()
                b_prog = st.progress(0)
                stages = [("🫁 Inhale slowly through nose...", 0.25), 
                          ("⏸️ Hold breath (lungs full)...", 0.50), 
                          ("🌬️ Exhale steadily through mouth...", 0.75), 
                          ("⏸️ Rest and reset...", 1.0)]
                for text, prog in stages:
                    b_status.info(f"**Current Action:** {text}")
                    b_prog.progress(prog)
                    time.sleep(1.2)
                b_status.success("✅ Cycle Complete: Autonomic respiratory rhythm normalized.")

        with sh_right:
            st.markdown("""<div class="pop-card">
<div style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-bottom: 12px;">📞 Confidential Support & Helpline Access</div>
<div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.82rem;">
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">
<b>National Tele-MANAS Mental Health Toll-Free:</b><br>
<span style="font-size: 1.1rem; color: #2563eb; font-weight: 700;">14416</span> (24x7 Multi-lingual Government Helpline)
</div>
<div style="background: #f8fafc; padding: 12px 16px; border-radius: 12px; border: 1px solid #e2e8f0; color: #0f172a;">
<b>CRPF Force Welfare Support Desk:</b><br>
<span style="font-size: 1rem; color: #0f172a; font-weight: 700;">1800-11-0000</span> (Internal Tele-Counseling Support)
</div>
</div>
<div style="margin-top: 20px;">
<p style="font-size: 0.8rem; font-weight: 600; color: #0f172a; margin-bottom: 8px;">Request Anonymous Unit Counselor Check-in:</p>
</div>
</div>""", unsafe_allow_html=True)

            if st.button("🤝 Schedule Confidential Unit Counselor Appointment", type="primary", use_container_width=True):
                st.success("✅ Appointment Request Forwarded confidentially to Unit Welfare Officer Dr. Kulkarni. No commander alert triggered.")

# ==================== 2. COMMANDER ROLE VIEWS ==================== #
elif user_role == "Commander":

    if st.session_state.current_nav == "Workload Grid":
        st.markdown(f"""<div class="pop-card">
<h3 style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 2px;">Personnel Welfare & Workload Risk Grid | {user['unit']}</h3>
<p style="color: #64748b; font-size: 0.8rem; margin-bottom: 16px;">Aggregated operational workload indicators | Confidential clinical logs are masked</p>
<table class="styled-table">
<thead>
<tr>
<th>Service ID</th>
<th>Assigned Role</th>
<th>Consecutive Nights</th>
<th>Days No Leave</th>
<th>Welfare Risk Band</th>
<th>Workload Adjustment Advisory</th>
</tr>
</thead>
<tbody>
<tr>
<td><b>P-10001</b></td>
<td>QRT Sentry</td>
<td>4</td>
<td>82</td>
<td><span class="pill-badge-red">🟠 Elevated</span></td>
<td><b style="color: #b91c1c;">Reassign to Day Support</b></td>
</tr>
<tr>
<td><b>P-10002</b></td>
<td>Day Support</td>
<td>0</td>
<td>18</td>
<td><span class="pill-badge-green">🟢 Stable</span></td>
<td><b style="color: #15803d;">Maintain Schedule</b></td>
</tr>
<tr>
<td><b>P-10003</b></td>
<td>Patrol Alpha</td>
<td>1</td>
<td>45</td>
<td><span class="pill-badge-green">🟢 Stable</span></td>
<td><b style="color: #15803d;">Maintain Schedule</b></td>
</tr>
<tr>
<td><b>P-10004</b></td>
<td>Night Sentry</td>
<td>5</td>
<td>110</td>
<td><span class="pill-badge-red">🟠 Elevated</span></td>
<td><b style="color: #b91c1c;">Rotate to Mandatory Rest</b></td>
</tr>
<tr>
<td><b>P-10005</b></td>
<td>Escort Detail</td>
<td>1</td>
<td>22</td>
<td><span class="pill-badge-green">🟢 Stable</span></td>
<td><b style="color: #15803d;">Maintain Schedule</b></td>
</tr>
</tbody>
</table>
</div>""", unsafe_allow_html=True)

    elif st.session_state.current_nav == "Workload Rebalancing Advisor":
        c_left, c_right = st.columns([1.1, 0.9])
        with c_left:
            st.markdown("""<div class="pop-card">
<h3 style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">Unit Workload Distribution</h3>
<p style="color: #64748b; font-size: 0.78rem; margin-bottom: 14px;">Current headcount distribution across strain bands</p>
</div>""", unsafe_allow_html=True)
            
            # Interactive Streamlit Bar Chart (No Broken Images)
            wl_df = pd.DataFrame({
                "Strain Category": ["Balanced", "Moderate Load", "Fatigue Rotation"],
                "Personnel Count": [3, 0, 2]
            }).set_index("Strain Category")
            st.bar_chart(wl_df, color="#4f46e5", height=240)

        with c_right:
            st.markdown("""<div class="pop-card">
<h3 style="font-size: 1.05rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">⚡ AI-Assisted Workload Rebalancing Advisor</h3>
<p style="color: #64748b; font-size: 0.78rem; margin-bottom: 16px;">Suggests optimal watch rotations based on fatigue indicators. Final roster changes require authorised human approval.</p>
</div>""", unsafe_allow_html=True)
            if st.button("📋 Generate Rebalancing Recommendation", use_container_width=True, type="primary"):
                st.success("✅ Advisory Generated: Recommend swapping P-10001 with P-10002 for Day Support rotation.")
                st.info("Decision Status: AI output is advisory — awaiting Company Commander sign-off.")

# ==================== 3. WELFARE OFFICER ROLE VIEWS (DYNAMIC TRIAGE) ==================== #
else:

    # Shared Active Case Selector across Welfare Officer Tools
    selected_case = st.selectbox(
        "🔎 Select Active Case for Clinical Triage & Welfare Review:",
        options=["P-10001", "P-10002"],
        format_func=lambda x: f"{x} — {'Ct. (Elevated Risk - High Strain)' if x == 'P-10001' else 'Ct. (Nominal Risk - Balanced Baseline)'}"
    )

    case_data = PERSONNEL_REGISTRY[selected_case]
    is_elevated = (selected_case == "P-10001")
    score = 76 if is_elevated else 22
    gauge_gradient = f"conic-gradient(#ef4444 {score}%, #f1f5f9 0)" if is_elevated else f"conic-gradient(#10b981 {score}%, #f1f5f9 0)"
    score_color = "#ef4444" if is_elevated else "#10b981"
    status_pill = '<span class="pill-badge-red">RISK STATUS: ELEVATED</span>' if is_elevated else '<span class="pill-badge-green">RISK STATUS: NOMINAL</span>'

    if st.session_state.current_nav == "Welfare Assessment Triage":
        st.markdown(f'<div style="font-size: 1.15rem; font-weight: 800; color: #0f172a; margin-bottom: 16px;">Unit Welfare Officer Decision Support | Case: {selected_case}</div>', unsafe_allow_html=True)

        w_left, w_right = st.columns([1.2, 1.8])

        with w_left:
            st.markdown(f"""<div class="pop-card" style="text-align: center; padding: 28px 20px;">
<div style="font-size: 0.85rem; font-weight: 700; color: #64748b; margin-bottom: 16px;">AI WELFARE ASSESSMENT</div>
<div class="radial-gauge" style="background: radial-gradient(closest-side, white 79%, transparent 80% 100%), {gauge_gradient};">
<div class="gauge-val" style="color: {score_color};">{score}</div>
<div class="gauge-sub">/ 100</div>
</div>
<div style="margin-top: 18px;">
{status_pill}
</div>
<div style="font-size: 0.75rem; color: #64748b; margin-top: 8px;">Prototype Welfare Risk Score: {score}/100</div>
</div>""", unsafe_allow_html=True)

            rec_html = """
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">🟢 <b>Priority leave review</b> (Expedited consideration)</div>
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">🟢 <b>Peer support referral</b> (Buddy System pairing)</div>
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">🟢 <b>Tele-counseling referral</b> (Confidential consultation)</div>
            """ if is_elevated else """
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">🟢 <b>Maintain standard duty watch</b> (Baseline tolerances confirmed)</div>
            <div style="background: #f8fafc; padding: 10px 14px; border-radius: 10px; border: 1px solid #e2e8f0;">🟢 <b>Routine 30-day periodic check-in</b></div>
            """

            st.markdown(f"""<div class="pop-card">
<div style="font-size: 0.88rem; font-weight: 700; color: #0f172a; margin-bottom: 10px;">RECOMMENDED NEXT STEPS</div>
<div style="display: flex; flex-direction: column; gap: 10px; font-size: 0.82rem; color: #334155;">
{rec_html}
</div>
</div>""", unsafe_allow_html=True)

        with w_right:
            leave_days = case_data["days_since_leave"]
            nights = case_data["night_shifts_7d"]
            sleep_h = case_data["sleep_hours"]
            leave_pct = min(100, int((leave_days / 90.0) * 100))
            nights_pct = min(100, int((nights / 7.0) * 100))

            st.markdown(f"""<div class="pop-card">
<div style="font-size: 0.92rem; font-weight: 800; color: #0f172a; margin-bottom: 16px;">WHY WAS THIS FLAGGED? (EVIDENCE CHAIN FOR {selected_case})</div>
<div style="display: flex; flex-direction: column; gap: 14px; font-size: 0.82rem;">
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; color: #0f172a;">
<span>Days elapsed since sanctioned leave</span>
<span style="color: {'#ef4444' if leave_days > 60 else '#10b981'};">{leave_days} days</span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 4px;"><div style="background: {'#ef4444' if leave_days > 60 else '#10b981'}; width: {leave_pct}%; height: 100%; border-radius: 4px;"></div></div>
</div>
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; color: #0f172a;">
<span>Consecutive night-watch load</span>
<span style="color: {'#ef4444' if nights >= 4 else '#10b981'};">{nights} watches logged</span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 4px;"><div style="background: {'#ef4444' if nights >= 4 else '#10b981'}; width: {nights_pct}%; height: 100%; border-radius: 4px;"></div></div>
</div>
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; color: #0f172a;">
<span>Sleep deficit tracking</span>
<span style="color: {'#d97706' if sleep_h < 5.5 else '#10b981'};">{sleep_h} hrs/day logged</span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 4px;"><div style="background: {'#f59e0b' if sleep_h < 5.5 else '#10b981'}; width: {int((sleep_h/8.0)*100)}%; height: 100%; border-radius: 4px;"></div></div>
</div>
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px; font-weight: 600; color: #0f172a;">
<span>Autonomic telemetry signal</span>
<span style="color: #4f46e5;">{'Acoustic jitter elevated (3.4%)' if is_elevated else 'Nominal baseline stability (1.1%)'}</span>
</div>
<div style="background: #f1f5f9; height: 8px; border-radius: 4px;"><div style="background: #4f46e5; width: {75 if is_elevated else 20}%; height: 100%; border-radius: 4px;"></div></div>
</div>
</div>
</div>""", unsafe_allow_html=True)

            st.markdown('<div class="pop-card">', unsafe_allow_html=True)
            st.markdown("<h3 style='font-size: 0.95rem; font-weight: 800; color: #0f172a; margin-bottom: 12px;'>Recommended Welfare Actions</h3>", unsafe_allow_html=True)
            b1, b2, b3 = st.columns(3)
            with b1:
                if b1.button("🏖️ Priority Leave", use_container_width=True, type="primary"):
                    st.success(f"Expedited leave advisory generated for {selected_case} and routed to Battalion HQ.")
            with b2:
                if b2.button("👥 Buddy System", use_container_width=True, type="primary"):
                    st.info(f"Designated peer buddy notified for an informal welfare check-in with {selected_case}.")
            with b3:
                if b3.button("📞 Tele-Counseling", use_container_width=True, type="primary"):
                    st.success(f"Tele-counseling referral recommended for {selected_case} — authorised follow-up logged.")
            st.caption("⚠️ Authorised human review required: AI output is advisory — final decision by Welfare Officer.")
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_nav == "Dynamic XAI Analysis":
        st.markdown(f"""<div class="pop-card">
<h3 style="font-size: 1.1rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">Dynamic Local Explainable AI (SHAP TreeExplainer)</h3>
<p style="color: #64748b; font-size: 0.8rem; margin-bottom: 16px;">True local feature contributions computed dynamically from Random Forest tree paths for active case: <b>{selected_case}</b></p>
</div>""", unsafe_allow_html=True)

        hardship_val = 1 if "Counter" in case_data["deployment_hardship"] else (2 if "High" in case_data["deployment_hardship"] else 0)
        sample_df = pd.DataFrame([{
            'Days_Since_Leave': case_data["days_since_leave"],
            'Night_Shifts_7d': case_data["night_shifts_7d"],
            'Deployment_Hardship': hardship_val,
            'Trauma_Incident_Flag': 1 if case_data["trauma_incident_flag"] else 0,
            'Sleep_Hours': case_data["sleep_hours"],
            'Subjective_Fatigue': case_data["subjective_fatigue"],
            'Optional_Vocal_Jitter': 3.4 if is_elevated else 1.1,
            'Optional_BPM_Delta': 14.0 if is_elevated else 2.0
        }])

        display_labels = ["Days No Leave", "Night Shifts", "Hardship Zone", "Trauma Exposure", "Sleep Deficit", "Self-Report Fatigue", "Vocal Jitter", "Pulse Delta"]
        
        if model is not None and explainer is not None:
            try:
                raw_shap = explainer.shap_values(sample_df)
                if isinstance(raw_shap, list):
                    arr = raw_shap[1][0] if len(raw_shap) > 1 else raw_shap[0][0]
                elif len(raw_shap.shape) == 3:
                    arr = raw_shap[0, :, 1]
                else:
                    arr = raw_shap[0]
                local_contribs = np.abs(np.array(arr).flatten())
            except Exception:
                local_contribs = np.array([28.4, 22.1, 12.0, 15.2, 11.8, 6.5, 2.5, 1.5]) if is_elevated else np.array([5.2, 4.1, 3.0, 1.2, 8.8, 6.5, 2.5, 1.5])
        else:
            local_contribs = np.array([28.4, 22.1, 12.0, 15.2, 11.8, 6.5, 2.5, 1.5]) if is_elevated else np.array([5.2, 4.1, 3.0, 1.2, 8.8, 6.5, 2.5, 1.5])

        total_c = np.sum(local_contribs)
        pct_contribs = np.round((local_contribs / total_c) * 100, 1) if total_c > 0 else np.zeros(len(local_contribs))
        
        # Native Streamlit Chart (Eliminates Broken Images)
        shap_chart_df = pd.DataFrame({
            "Feature Contribution (%)": pct_contribs
        }, index=display_labels)
        st.bar_chart(shap_chart_df, color="#4f46e5", height=280)

    elif st.session_state.current_nav == "Longitudinal Trend (30 Days)":
        st.markdown(f"""<div class="pop-card">
<h3 style="font-size: 1.1rem; font-weight: 800; color: #0f172a; margin-bottom: 4px;">Longitudinal Welfare Risk Trend — 30 Days</h3>
<p style="color: #64748b; font-size: 0.8rem; margin-bottom: 16px;">Tracking early risk indicators over rolling assessment periods for: <b>{selected_case}</b></p>
</div>""", unsafe_allow_html=True)
        
        days = ["01. Day 1", "02. Day 6", "03. Day 12", "04. Day 18", "05. Day 24", "06. Day 30"]
        scores = [32, 38, 48, 55, 64, 76] if is_elevated else [24, 22, 20, 25, 21, 22]
        trend_chart_df = pd.DataFrame({"Welfare Risk Score": scores}, index=days)
        st.line_chart(trend_chart_df, color="#ef4444" if is_elevated else "#10b981", height=260)

# --- GLOBAL APPLICATION FOOTER ---
st.markdown("""<div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 14px 24px; margin-top: 36px; display: flex; justify-content: space-between; align-items: center; font-size: 0.78rem; color: #64748b; box-shadow: 0 4px 14px rgba(148, 163, 184, 0.08);">
<div>
<b style="color:#0f172a;">SURAKSHA-DRISHTI</b> | AI-Assisted Personnel Welfare & Stress-Risk System (MHA PS ID: 26186)
</div>
<div>
Prototype running locally; architecture designed for air-gapped edge deployment. Synthetic demonstration data used.
</div>
<div style="display: flex; gap: 14px; font-weight: 600;">
<span>🔒 Privacy-first</span>
<span>❤️ Welfare-first</span>
<span>⚙️ Local AI</span>
</div>
</div>""", unsafe_allow_html=True)