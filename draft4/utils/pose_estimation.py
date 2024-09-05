import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

model_path = r"E:\Projects\SIH\draft4\Models\pose_landmarker_lite.task" 
# Make sure this path is correct for your system!

base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True 
)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draws detected landmarks on an image."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def resize_window(frame):
    """Resizes the image frame to a fixed window size."""
    window_width = 800
    window_height = 600
    original_height, original_width = frame.shape[:2]
    scale_width = window_width / original_width
    scale_height = window_height / original_height
    scale = min(scale_width, scale_height)
    
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized_image = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

def process_frame_for_pose(frame):
    """Processes a frame to detect pose landmarks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    landmarks = None
    annotated_image = frame

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]  # Get landmarks from first person detected
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        
    resized_image = resize_window(annotated_image)
    return landmarks, resized_image

def get_landmark_coordinates(landmarks):
    """Returns a dictionary of landmark coordinates."""
    if not landmarks:
        return None
    return {
        'left_shoulder': [landmarks[11].x, landmarks[11].y, landmarks[11].z],
        'left_elbow': [landmarks[13].x, landmarks[13].y, landmarks[13].z],
        'left_wrist': [landmarks[15].x, landmarks[15].y, landmarks[15].z],
        'right_shoulder': [landmarks[12].x, landmarks[12].y, landmarks[12].z],
        'right_elbow': [landmarks[14].x, landmarks[14].y, landmarks[14].z],
        'right_wrist': [landmarks[16].x, landmarks[16].y, landmarks[16].z],
        'left_hip': [landmarks[23].x, landmarks[23].y, landmarks[23].z],
        'left_knee': [landmarks[25].x, landmarks[25].y, landmarks[25].z],
        'left_ankle': [landmarks[27].x, landmarks[27].y, landmarks[27].z],
        'right_hip': [landmarks[24].x, landmarks[24].y, landmarks[24].z],
        'right_knee': [landmarks[26].x, landmarks[26].y, landmarks[26].z],
        'right_ankle': [landmarks[28].x, landmarks[28].y, landmarks[28].z],
    }
