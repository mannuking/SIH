import cv2
import tempfile
import os
import numpy as np  # Import numpy for angle calculations
from utils.pose_estimation import process_frame_for_pose
from utils.llava_interaction import get_feedback_from_llava
import streamlit as st
import mediapipe as mp

class VideoProcessor:
    def __init__(self, workout_plan):
        self.workout_plan = workout_plan

    def process_video(self):
        # Record a video section
        recorded_video_path = self.record_video()

        # Analyze the recorded video
        feedback = self.analyze_video(recorded_video_path)

        # Display feedback (you can customize this part)
        st.write("## Workout Analysis:")
        st.write(feedback)

        # Cleanup (optional, delete the temporary video file)
        os.remove(recorded_video_path)

    def record_video(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        out = cv2.VideoWriter(
            temp_video_file.name,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20.0,
            (frame_width, frame_height),
        )

        st.write("Recording... Press 'q' to stop.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            cv2.imshow("Recording", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return temp_video_file.name

    def analyze_video(self, video_path):
        # Implement video analysis logic here
        cap = cv2.VideoCapture(video_path)
        feedback = ""
        rep_count = 0
        set_count = 0
        is_exercising = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process each frame for analysis (e.g., pose estimation)
            pose_landmarks, _ = process_frame_for_pose(frame, self.workout_plan["name"])

            # Example: Simple rep counting based on a condition (customize for your workout)
            if pose_landmarks:
                # Extract relevant angles or features from pose_landmarks
                if self.workout_plan["name"] == "Squats":
                    angle = self.calculate_squat_angle(pose_landmarks)
                    if angle < 110 and not is_exercising:  # Example threshold
                        is_exercising = True
                        rep_count += 1
                elif self.workout_plan["name"] == "Pushups":
                    angle = self.calculate_pushup_angle(pose_landmarks)
                    if angle < 90 and not is_exercising:  # Example threshold
                        is_exercising = True
                        rep_count += 1
                elif self.workout_plan["name"] == "Bicep Curls":
                    angle = self.calculate_bicep_curl_angle(pose_landmarks)
                    if angle > 160 and not is_exercising:  # Example threshold
                        is_exercising = True
                        rep_count += 1
                else:
                    is_exercising = False

                # Check for set completion (example: reps per set based on workout plan)
                if rep_count == self.workout_plan["target_reps"]:
                    set_count += 1
                    rep_count = 0

                # Generate feedback based on analysis (e.g., using LLaVA)
                llava_feedback = get_feedback_from_llava(frame, self.workout_plan["name"])
                feedback += llava_feedback 
                feedback += f"\nReps: {rep_count}, Sets: {set_count}\n\n"

        cap.release()
        return feedback

    def calculate_angle(self, landmark1, landmark2, landmark3):
        """Calculates the angle between three landmarks.

        Args:
            landmark1: The first landmark (e.g., shoulder).
            landmark2: The middle landmark (e.g., elbow).
            landmark3: The third landmark (e.g., wrist).

        Returns:
            The angle in degrees.
        """

        a = np.array([landmark1.x, landmark1.y])  # First
        b = np.array([landmark2.x, landmark2.y])  # Mid
        c = np.array([landmark3.x, landmark3.y])  # End

        # Calculate vectors
        ba = a - b
        bc = c - b

        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(cosine_angle))

        return angle

        
    
    def calculate_pushup_angle(self, landmarks):
        """Calculates the pushup angle (shoulder-elbow-wrist).

        Args:
            landmarks: The detected pose landmarks.

        Returns:
            The pushup angle in degrees.
        """

        shoulder = landmarks[mp.pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp.pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp.pose.PoseLandmark.LEFT_WRIST]

        return self.calculate_angle(shoulder, elbow, wrist)

    def calculate_bicep_curl_angle(self, landmarks):
        """Calculates the bicep curl angle (shoulder-elbow-wrist).

        Args:
            landmarks: The detected pose landmarks.

        Returns:
            The bicep curl angle in degrees.
        """

        shoulder = landmarks[mp.pose.PoseLandmark.LEFT_SHOULDER]
        elbow = landmarks[mp.pose.PoseLandmark.LEFT_ELBOW]
        wrist = landmarks[mp.pose.PoseLandmark.LEFT_WRIST]

        return self.calculate_angle(shoulder, elbow, wrist)
    
    def calculate_squat_angle(self, landmarks):
        """Calculates the squat angle (hip-knee-ankle).

        Args:
            landmarks: The detected pose landmarks.

        Returns:
            The squat angle in degrees.
        """

        hip = landmarks[mp.pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp.pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp.pose.PoseLandmark.LEFT_ANKLE]

        return self.calculate_angle(hip, knee, ankle)
    
    
