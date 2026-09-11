"""
SURAKSHA-DRISHTI: Acoustic Processing Engine (Voice Pitch Variability)
Aligned with MHA CRPF Problem Statement ID: 26186

Non-Diagnostic Notice:
This module extracts an auxiliary pitch-period perturbation proxy (acoustic variability).
It serves as a contextual physiological strain indicator and does NOT provide
definitive clinical, psychological, or medical stress diagnoses.
"""

import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks
import time

def extract_voice_metrics(duration=5, fs=44100):
    print(f"[*] Recording audio snippet ({duration}s) for acoustic strain proxy analysis...")
    
    try:
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()
    except Exception as e:
        print(f"[!] Audio input hardware error: {e}")
        return {
            "vocal_jitter": 1.2,
            "vocal_stress_score": 30.0,
            "status": "Hardware Sensor Inaccessible (Default Baseline Applied)"
        }

    audio = recording.flatten()
    frame_size = int(fs * 0.04)  # 40ms window
    hop_size = int(fs * 0.02)    # 20ms overlap
    
    pitches = []
    for i in range(0, len(audio) - frame_size, hop_size):
        chunk = audio[i:i + frame_size]
        # Energy threshold check to ignore silent frames
        if np.sqrt(np.mean(chunk**2)) > 0.01:
            chunk_corr = np.correlate(chunk, chunk, mode='full')[len(chunk)//2:]
            peaks, _ = find_peaks(chunk_corr[int(fs/350):int(fs/70)], distance=15)
            if len(peaks) > 0:
                pitches.append(fs / (peaks[0] + int(fs/350)))

    if len(pitches) > 5:
        pitch_variability = float(np.mean(np.abs(np.diff(pitches))) / np.mean(pitches) * 100)
    else:
        pitch_variability = 1.2

    # Contextual scalar mapping for multi-modal feature vector
    acoustic_strain_proxy = float(np.clip((pitch_variability - 0.8) * 35.0, 10.0, 95.0))
    
    # Scientifically responsible non-diagnostic classification
    if pitch_variability > 2.8:
        status = "Acoustic Variability: ELEVATED (Contextual Strain Signal)"
    else:
        status = "Acoustic Variability: WITHIN BASELINE TOLERANCE"

    print("\n--- ACOUSTIC STRAIN PROXY REPORT ---")
    print(f"Voice Pitch Variability : {pitch_variability:.2f}%")
    print(f"Acoustic Strain Proxy   : {acoustic_strain_proxy:.1f}/100")
    print(f"Indicator Status        : {status}")
    print("------------------------------------\n")

    return {
        "vocal_jitter": pitch_variability,
        "vocal_stress_score": acoustic_strain_proxy,
        "status": status
    }

if __name__ == "__main__":
    extract_voice_metrics(duration=5)