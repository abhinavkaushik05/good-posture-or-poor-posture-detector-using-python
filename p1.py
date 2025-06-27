import cv2
import mediapipe as mp
import numpy as np
import time
from playsound import playsound 
import os

# Initialize Mediapipe pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0) #sets up mediapipe pose to detects ypur body

# Initialize variables (missing!)
is_calibrated = False
calibration_frames = 0
calibration_shoulder_angels = []
calibration_neck_angels = []#these are used to calibrate your posture during the first few second
last_alert_time = 0
alert_cooldown = 5  # in seconds
sound_file = "alert.mp3"  # change this to your sound file path

# Main loop (fix whilecap)
while cap.isOpened():
    ret, frame = cap.read()#keeps reading live video frames until you press 'q'.
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)#detects your pose landmarks (like shoulders, ears).

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Step 2: Pose detection
        # Extract key body landmarks
        left_shoulder = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0])
        )
        right_shoulder = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0])
        )
        left_ear = (
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0])
        )
        right_ear = (
            int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
            int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0])#{left/right shoulders left/right ears}
        )

        # STEP #: Angle calculation
        def calculate_angle(a, b, c):
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)

        def draw_angle(frame, a, b, c, angle, color):
            cv2.putText(frame, f"{int(angle)} deg", (b[0] + 10, b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))# this function finds the angle between three points. you use it to calculate: shoulder angle , neck angle

        # Step 1: Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angels.append(shoulder_angle)
            calibration_neck_angels.append(neck_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating...{calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angels) - 10
            neck_threshold = np.mean(calibration_neck_angels) - 10
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")# takes your normal postureas a baseline. averages the shoulder & neck angles. after 30 frames, it saves a thershold (ideal posture minus 10 degrees)

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))#drwa your skeleton and displays the angles on screen.

        # Step 4: Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor Posture detected! Please sit up straight.")
                    if os.path.exists(sound_file):
                        playsound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)# if your current angle is worse than the calibrated threshold , it says : "poor posture"(red color), if posture is okay: shows " good posture"(green color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()# shows the webcan window. press 'q' to quit.