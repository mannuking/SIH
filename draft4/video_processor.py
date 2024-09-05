import cv2
import tempfile
import os
import numpy as np
from utils.pose_estimation import process_frame_for_pose
from utils.llava_interaction import get_feedback_from_llava
import streamlit as st
import mediapipe as mp
import threading
import time
import pyttsx3
import logging

# Configure logging
logging.basicConfig(
    filename="workout_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

mp_pose = mp.solutions.pose


class VideoProcessor:
    def __init__(self):
        self.recording = False
        self.background_recorder = None
        self.current_workout = None
        self.rep_count = 0
        self.set_count = 0
        self.is_exercising = False
        self.tts_engine = pyttsx3.init()
        self.workout_start_time = None
        self.ai_feedback = ""
        self.show_stop_button = False
        logging.info("VideoProcessor initialized.")

    def process_video(self):
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        st.write("Recording...")

        frame_placeholder = st.empty()
        feedback_placeholder = st.empty()
        # Start recording immediately
        self.start_background_recording()
        start_time = time.time()

        while st.session_state.workout_active:
            ret, frame = cap.read()
            if not ret:
                break

            pose_landmarks, annotated_frame = process_frame_for_pose(frame)

            if pose_landmarks:
                if not self.current_workout:
                    self.current_workout = self.recognize_workout(
                        pose_landmarks
                    )
                    self.workout_start_time = time.time()

                self.process_workout(pose_landmarks)

            # Display workout data on frame
            feedback_text = f"{self.current_workout}: Reps: {self.rep_count}, Sets: {self.set_count}\n"
            cv2.putText(
                annotated_frame,
                feedback_text,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            frame_placeholder.image(annotated_frame, channels="BGR")

            # Show Stop Button After 15 Seconds
            if time.time() - start_time >= 15 and not self.show_stop_button:
                self.show_stop_button = True

            if cv2.waitKey(1) & 0xFF == ord("q"):  # Emergency stop
                break

        cap.release()
        cv2.destroyAllWindows()

        if self.background_recorder:
            self.background_recorder.join()
            recorded_video_path = self.background_recorder.video_path
            self.background_recorder = None
            llava_feedback = self.analyze_video_with_llava(
                recorded_video_path
            )
            # self.provide_audio_feedback(llava_feedback)
            self.ai_feedback = llava_feedback
            os.remove(recorded_video_path)  #it is made to ensure data privacy of users


    def recognize_workout(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]

        squat_angle = self.calculate_angle(hip, knee, ankle)
        arm_angle = self.calculate_angle(shoulder, elbow, wrist)

        if squat_angle < 120:
            return "Squats"
        elif arm_angle < 90:
            return "Push-ups"
        else:
            return "Unknown"

    def process_workout(self, landmarks):
        logging.debug(f"Current workout: {self.current_workout}")

        if self.current_workout == "Squats":
            angle = self.calculate_angle(landmarks[23], landmarks[25], landmarks[27])
            threshold = 110 
        elif self.current_workout == "Push-ups": 
            angle = self.calculate_angle(landmarks[11], landmarks[13], landmarks[15])
            threshold = 90 
        else:
            return

        logging.debug(f"Calculated angle: {angle}")
        logging.debug(f"is_exercising: {self.is_exercising}")

        if angle < threshold and not self.is_exercising:
            self.is_exercising = True
            self.rep_count += 1
            logging.debug(f"Rep started. Rep count: {self.rep_count}")
            
        elif angle > threshold + 40 and self.is_exercising: 
            self.is_exercising = False
            logging.debug("Rep ended.")

        if self.rep_count == 10: 
            self.set_count += 1
            self.rep_count = 0
            

    def start_background_recording(self):
        self.recording = True
        self.background_recorder = BackgroundRecorder()
        self.background_recorder.start()

    def analyze_video_with_llava(self, video_path):
        cap = cv2.VideoCapture(video_path)
        feedback = ""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            llava_feedback = get_feedback_from_llava(frame, self.current_workout)
            feedback += llava_feedback + "\n"
        cap.release()
        return feedback
    
    def provide_audio_feedback(self, feedback):
        self.tts_engine.say(feedback)
        self.tts_engine.runAndWait()

    # def process_set_completion(self):
    #     if self.background_recorder:
    #         self.background_recorder.stop()
    #         recorded_video_path = self.background_recorder.video_path
    #         # Reset the background recorder
    #         self.background_recorder = None  
    #         llava_feedback = self.analyze_video_with_llava(recorded_video_path)
    #         self.provide_audio_feedback(llava_feedback)
    #         self.ai_feedback = llava_feedback  # Store feedback for text visualization
    #         os.remove(recorded_video_path)

    def calculate_angle(self, landmark1, landmark2, landmark3):
        a = np.array([landmark1.x, landmark1.y])
        b = np.array([landmark2.x, landmark2.y])
        c = np.array([landmark3.x, landmark3.y])

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle 


class BackgroundRecorder(threading.Thread):
    def __init__(self):
        super(BackgroundRecorder, self).__init__()
        self.frame_width = 640
        self.frame_height = 480
        self.video_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
        self.out = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (self.frame_width, self.frame_height))
        self.stop_flag = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        while (
            not self.stop_flag.is_set() and time.time() - start_time < 15
        ):  # Record for 15 seconds
            ret, frame = cap.read()
            if not ret:
                break
            self.out.write(frame)
        self.out.release()
        cap.release()

    def stop(self):
        self.stop_flag.set()
