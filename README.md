# good-posture-or-poor-posture-detector-using-python
 Posture Corrector
What it does:

-Uses your webcam to detect upper‑body posture via Mediapipe.
-Calibrates on your “good” posture (30 frames), sets personal thresholds (avg – 10°).
-Tracks shoulder & neck angles in real time.

Feedback:

-Displays “Good Posture” or “Poor Posture” with angle values and skeleton overlay.
-Plays an alert sound (alert.mp3) every 5 s if posture dips below threshold.

Usage:

1. pip install opencv-python mediapipe numpy playsound

2. Place alert.mp3 in root.

3. Run python posture_corrector.py.

4. Sit straight for calibration.

5. Adjust as needed (threshold offset, cooldown, calibration frames).

6. Press q to quit.
