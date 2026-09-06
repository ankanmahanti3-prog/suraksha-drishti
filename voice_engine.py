import numpy as np
import sounddevice as sd
from scipy.signal import find_peaks

def record_and_analyze_voice(duration=5, fs=44100):
    print("\n--- INITIATING 5-SECOND VOICE CHECK-IN ---")
    print("When recording starts, clearly speak a check-in phrase:")
    print("Example: 'Constable Kumar, ID 4082, reporting for duty.'\n")
    
    # Record audio directly from laptop mic
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()  # Wait until the 5 seconds complete
    print(">> Audio capture complete. Processing acoustic biomarkers...")

    audio = recording.flatten()
    
    # Check if there was actual speech (avoid dead silence)
    rms_energy = np.sqrt(np.mean(audio**2))
    if rms_energy < 0.005:
        print(">> Warning: Microphone input too quiet. Please speak closer to the mic.")
        return {"jitter_pct": 1.0, "stress_index": 50, "status": "Low Volume Sample"}

    # Autocorrelation to extract fundamental frequency (pitch)
    corr = np.correlate(audio, audio, mode='full')
    corr = corr[len(corr)//2:]

    # Pitch search window (70 Hz to 350 Hz for human speech)
    min_lag = int(fs / 350)
    max_lag = int(fs / 70)
    
    peaks, _ = find_peaks(corr[min_lag:max_lag], distance=20)
    
    if len(peaks) > 0:
        peak_lag = peaks[np.argmax(corr[min_lag + peaks])] + min_lag
        pitch_hz = fs / peak_lag
    else:
        pitch_hz = 120.0  # Fallback baseline pitch

    # Calculate frame-by-frame pitch variations (Jitter proxy)
    frame_size = int(fs * 0.04)  # 40ms window
    hop_size = int(fs * 0.02)
    pitches = []

    for i in range(0, len(audio) - frame_size, hop_size):
        frame = audio[i:i + frame_size]
        if np.sqrt(np.mean(frame**2)) > 0.01:
            frame_corr = np.correlate(frame, frame, mode='full')
            frame_corr = frame_corr[len(frame_corr)//2:]
            f_peaks, _ = find_peaks(frame_corr[min_lag:max_lag], distance=15)
            if len(f_peaks) > 0:
                lag = f_peaks[np.argmax(frame_corr[min_lag + f_peaks])] + min_lag
                pitches.append(fs / lag)

    # Compute vocal jitter (instability)
    if len(pitches) > 5:
        pitch_diffs = np.abs(np.diff(pitches))
        jitter_pct = (np.mean(pitch_diffs) / np.mean(pitches)) * 100
    else:
        jitter_pct = 1.2  # Normal relaxed range baseline

    # Map jitter to a 0 - 100 Acoustic Stress Index
    # Normal voice jitter is 0.5% - 2.0%. Fatigued/stressed voice rises above 3.5%
    stress_score = float(np.clip((jitter_pct - 0.8) * 35.0, 10, 95))

    print("\n==============================")
    print(f">> Mean Fundamental Pitch: {pitch_hz:.1f} Hz")
    print(f">> Vocal Jitter Index: {jitter_pct:.2f}%")
    print(f">> Acoustic Stress Score: {stress_score:.1f} / 100")
    
    if stress_score > 65:
        print(">> Vocal Strain: ELEVATED (Micro-tremors detected)")
    else:
        print(">> Vocal Strain: STABLE / NORMAL")
    print("==============================\n")

    return {
        "pitch_hz": pitch_hz,
        "jitter_pct": jitter_pct,
        "stress_score": stress_score
    }

if __name__ == "__main__":
    record_and_analyze_voice()