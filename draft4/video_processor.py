import cv2
import tempfile
import os
from utils.pose_estimation import process_frame_for_pose
from utils.llava_interaction import get_feedback_from_llava
import streamlit as st

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
        # ... (Implement your video analysis logic here)
        # You can use LLaVA or other methods for analysis
        cap = cv2.VideoCapture(video_path)
        feedback = ""

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # ... (Process each frame for analysis, e.g., pose estimation)
            pose_landmarks, _ = process_frame_for_pose(frame, self.workout_plan["name"])

            # ... (Generate feedback based on analysis)
            if pose_landmarks:
                feedback += get_feedback_from_llava(frame, self.workout_plan["name"])
                feedback += "\n"  # Add a newline for readability

        cap.release()
        return feedback
