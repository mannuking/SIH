import cv2
import tempfile
import os
import numpy as np
from utils.pose_estimation import process_frame_for_pose
from utils.llava_interaction import get_feedback_from_llava
import streamlit as st
import mediapipe as mp
import threading  # For background recording
import time  # For frame rate control

# Check if the pose module is available (for debugging)
if not hasattr(mp, 'solutions') or not hasattr(mp.solutions, 'pose'):
    raise ImportError("MediaPipe pose module not found. Please check your installation.")

mp_pose = mp.solutions.pose

class VideoProcessor:
    def __init__(self, workout_plan):
        self.workout_plan = workout_plan
        self.recording = False  # Flag to indicate if recording is in progress
        self.background_recorder = None  # Thread for background recording

    def process_video(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        st.write("Recording... Press 'q' to stop.")

        rep_count = 0
        set_count = 0
        is_exercising = False
        feedback = ""

        # Create a placeholder for the video frame
        frame_placeholder = st.empty()

        # Set the desired frame rate (e.g., 10 FPS)
        desired_fps = 10
        frame_interval = 1 / desired_fps

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for pose estimation
            pose_landmarks, annotated_frame = process_frame_for_pose(frame, self.workout_plan["name"])

            # Real-time feedback (e.g., rep counting for squats)
            if pose_landmarks:
                if self.workout_plan["name"] == "Squats":
                    angle = self.calculate_squat_angle(pose_landmarks)
                    if angle < 110 and not is_exercising:
                        is_exercising = True
                        rep_count += 1
                        if rep_count == 1:  # Start recording after the first rep
                            self.start_background_recording(cap, frame_width, frame_height)
                    elif angle > 150 and is_exercising:
                        is_exercising = False
                # ... (Add logic for other exercises)

                # Check for set completion
                if rep_count == self.workout_plan["target_reps"]:
                    set_count += 1
                    rep_count = 0

                # Display real-time feedback
                feedback = f"Reps: {rep_count}, Sets: {set_count}\n"
                cv2.putText(annotated_frame, feedback, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update the video frame in the placeholder
            frame_placeholder.image(annotated_frame, channels="BGR")

            # Control frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_interval:
                time.sleep(frame_interval - elapsed_time)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Stop background recording
        if self.background_recorder:
            self.background_recorder.join()

        # Analyze the recorded video with LLaMA (if recording happened)
        if self.recording:
            recorded_video_path = self.background_recorder.video_path
            llava_feedback = self.analyze_video_with_llama(recorded_video_path)
            st.write("## Workout Analysis:")
            st.write(llava_feedback)
            os.remove(recorded_video_path)  # Cleanup

    def start_background_recording(self, cap, frame_width, frame_height):
        self.recording = True
        self.background_recorder = BackgroundRecorder(cap, frame_width, frame_height)
        self.background_recorder.start()

    def analyze_video_with_llama(self, video_path):
        cap = cv2.VideoCapture(video_path)
        feedback = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            llava_feedback = get_feedback_from_llava(frame, self.workout_plan["name"])
            feedback += llava_feedback + "\n"
        cap.release()
        return feedback


    def calculate_angle(self, landmark1, landmark2, landmark3):
        """Calculates the angle between three landmarks (MediaPipe Tasks format)."""

        a = np.array([landmark1.x, landmark1.y])
        b = np.array([landmark2.x, landmark2.y])
        c = np.array([landmark3.x, landmark3.y])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

    def calculate_pushup_angle(self, landmarks):
        shoulder = landmarks[11]  # LEFT_SHOULDER in MediaPipe Tasks
        elbow = landmarks[13]  # LEFT_ELBOW
        wrist = landmarks[15]  # LEFT_WRIST
        return self.calculate_angle(shoulder, elbow, wrist)

    def calculate_bicep_curl_angle(self, landmarks):
        shoulder = landmarks[11]  # LEFT_SHOULDER in MediaPipe Tasks
        elbow = landmarks[13]  # LEFT_ELBOW
        wrist = landmarks[15]  # LEFT_WRIST
        return self.calculate_angle(shoulder, elbow, wrist)

    def calculate_squat_angle(self, landmarks):
        hip = landmarks[23]  # LEFT_HIP in MediaPipe Tasks
        knee = landmarks[25]  # LEFT_KNEE
        ankle = landmarks[27]  # LEFT_ANKLE
        return self.calculate_angle(hip, knee, ankle)


class BackgroundRecorder(threading.Thread):
    def __init__(self, cap, frame_width, frame_height):
        super(BackgroundRecorder, self).__init__()
        self.cap = cap
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        self.out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frame_width, frame_height))

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.out.write(frame)
        self.out.release()
