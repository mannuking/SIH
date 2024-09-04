import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def process_frame_for_pose(frame, exercise_type):
    """Processes a frame to detect pose landmarks and annotate the frame."""

    # Recolor image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make detection
    results = pose.process(image)
    #I HAVE MADE THE CHANGES

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    pose_landmarks = None
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Exercise-specific analysis (e.g., angle calculations)
        # ... (Add logic for different exercise types here)

    return pose_landmarks, image 
