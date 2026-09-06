import cv2
import numpy as np
from scipy.signal import butter, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.75, highcut=3.0, fs=30.0, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def run_10s_pulse_scan():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    fps = 30.0
    duration_sec = 10
    total_frames = int(fps * duration_sec)
    
    green_signals = []
    print("\n--- INITIATING 10-SECOND BIOMETRIC SCAN ---")
    print("Keep your head still and look directly at the camera.\n")
    
    frame_count = 0
    
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))
        
        # Default fallback value if face momentarily drops
        current_g = green_signals[-1] if len(green_signals) > 0 else 100.0
        
        for (x, y, w_box, h_box) in faces:
            # Forehead Box
            fh_x1 = int(x + w_box * 0.28)
            fh_x2 = int(x + w_box * 0.72)
            fh_y1 = int(y + h_box * 0.08)
            fh_y2 = int(y + h_box * 0.22)
            
            roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]
            if roi.size > 0:
                current_g = roi[:, :, 1].mean()
                cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 255, 0), 2)
            break
            
        green_signals.append(current_g)
        frame_count += 1
        
        # On-screen HUD Countdown
        progress = int((frame_count / total_frames) * 100)
        remaining = duration_sec - int(frame_count / fps)
        
        cv2.putText(frame, f"TACTICAL SCAN: {progress}%", (30, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Time Remaining: {remaining}s", (30, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Visual Progress Bar
        cv2.rectangle(frame, (30, 100), (30 + int(progress * 3.5), 115), (0, 255, 0), -1)
        cv2.rectangle(frame, (30, 100), (380, 115), (255, 255, 255), 2)
        
        cv2.imshow("Roll-Call Biometric Kiosk", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(green_signals) < 150:
        print("Scan aborted early.")
        return None
        
    # --- SIGNAL FILTERING & FFT ---
    raw_signal = np.array(green_signals)
    # Detrend to remove lighting drift
    detrended = raw_signal - np.mean(raw_signal)
    
    # 0.75 Hz (45 bpm) to 3.0 Hz (180 bpm) Bandpass
    filtered = bandpass_filter(detrended, lowcut=0.75, highcut=3.0, fs=fps)
    
    # Fast Fourier Transform to find peak frequency
    fft_vals = np.abs(np.fft.rfft(filtered))
    fft_freqs = np.fft.rfftfreq(len(filtered), d=1.0/fps)
    
    # Keep only valid heart frequencies
    valid_idx = np.where((fft_freqs >= 0.75) & (fft_freqs <= 3.0))
    valid_freqs = fft_freqs[valid_idx]
    valid_fft = fft_vals[valid_idx]
    
    peak_freq = valid_freqs[np.argmax(valid_fft)]
    bpm = peak_freq * 60.0
    
    print("\n==============================")
    print(f">> SCAN COMPLETE!")
    print(f">> Estimated Pulse: {bpm:.1f} BPM")
    
    if bpm > 95:
        print(">> Autonomic Stress Status: HIGH (Tachycardia / Elevated Strain)")
    elif bpm < 60:
        print(">> Autonomic Stress Status: LOW / ATHLETIC REST")
    else:
        print(">> Autonomic Stress Status: NORMAL / COMBAT READY")
    print("==============================\n")
    
    return bpm

if __name__ == "__main__":
    run_10s_pulse_scan()