import cv2
import numpy as np

# Load OpenCV's built-in pre-trained face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
print("Starting camera... Press 'q' on the video window to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip horizontally for natural mirror view
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)

        # Forehead Region of Interest (ROI): Upper center area of the face box
        fh_x1 = int(x + w * 0.28)
        fh_x2 = int(x + w * 0.72)
        fh_y1 = int(y + h * 0.08)
        fh_y2 = int(y + h * 0.22)

        forehead_roi = frame[fh_y1:fh_y2, fh_x1:fh_x2]

        if forehead_roi.size > 0:
            # Extract mean green intensity (channel 1 = Green, best absorption for rPPG)
            mean_green = forehead_roi[:, :, 1].mean()

            # Green box showing active pulse capture
            cv2.rectangle(frame, (fh_x1, fh_y1), (fh_x2, fh_y2), (0, 255, 0), 2)
            cv2.putText(frame, f"rPPG Pulse ROI (G: {mean_green:.1f})", 
                        (fh_x1 - 20, fh_y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Roll-Call Welfare Kiosk - Pulse Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()