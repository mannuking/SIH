import cv2
import mediapipe as mp
import numpy as np

# Import MediaPipe Tasks 
from mediapipe.tasks import python
from mediapipe.tasks.python import vision 
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


# Specify the path to the model
model_path = r"E:\Projects\SIH\draft4\Models\pose_landmarker_lite.task"

# Initialize MediaPipe PoseLandmarker with options
base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True
)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    bc = c - b
    
    dot_product = np.dot(ab, bc)
    magnitude_ab = np.linalg.norm(ab)
    magnitude_bc = np.linalg.norm(bc)
    
    angle = np.arccos(dot_product / (magnitude_ab * magnitude_bc))
    angle = np.degrees(angle)
    
    return angle


def resize_window(frame):
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


def process_frame_for_pose(frame, exercise_type):
    push_up_count = 0
    is_push_up = False
    squat_count = 0
    is_squatting = False
    threshold_push_angle = 100
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe image format (required for MediaPipe Tasks)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Run pose detection using the detector object 
    detection_result = detector.detect(mp_image)

    landmarks = None
    annotated_image = frame

    if detection_result.pose_landmarks:
        landmarks = detection_result.pose_landmarks[0]  # Get the first detected pose

        left_shoulder = [landmarks[11].x, landmarks[11].y, landmarks[11].z]
        left_elbow = [landmarks[13].x, landmarks[13].y, landmarks[13].z]
        left_wrist = [landmarks[15].x, landmarks[15].y, landmarks[15].z]

        right_shoulder = [landmarks[12].x, landmarks[12].y, landmarks[12].z]
        right_elbow = [landmarks[14].x, landmarks[14].y, landmarks[14].z]
        right_wrist = [landmarks[16].x, landmarks[16].y, landmarks[16].z]

        left_hip = [landmarks[23].x, landmarks[23].y, landmarks[23].z]
        left_knee = [landmarks[25].x, landmarks[25].y, landmarks[25].z]
        left_ankle = [landmarks[27].x, landmarks[27].y, landmarks[27].z]

        right_hip = [landmarks[24].x, landmarks[24].y, landmarks[24].z]
        right_knee = [landmarks[26].x, landmarks[26].y, landmarks[26].z]
        right_ankle = [landmarks[28].x, landmarks[28].y, landmarks[28].z]
        
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        threshold_squat_angle = 90
        if left_knee_angle < threshold_squat_angle:
            if not is_squatting:
                is_squatting = True
        else:
            if is_squatting:
                is_squatting = False
                squat_count += 1

        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_elbow_angle < threshold_push_angle: 
            if not is_push_up:
                is_push_up = True
        else:
            if is_push_up:
                push_up_count += 1
                is_push_up = False

        # Use the draw_landmarks_on_image function for MediaPipe Tasks
        annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
        cv2.putText(annotated_image, f"Push Up Count: {push_up_count}", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(annotated_image, f"Squat Count: {squat_count}", (3, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
    resized_image = resize_window(annotated_image)
    return landmarks, resized_image
