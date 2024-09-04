import streamlit as st
import cv2
from utils.pose_estimation import process_frame_for_pose  # Assuming you have these utils files
from utils.llava_interaction import get_feedback_from_llava
from video_processor import VideoProcessor
from workout_plans import get_workout_plans

st.title("WorkoutAI - AI Fitness Coach")

# Sidebar for workout selection
workout_plans = get_workout_plans()
selected_workout = st.sidebar.selectbox(
    "Choose a workout plan:", list(workout_plans.keys())
)

# Main content
st.write(f"Selected workout: {selected_workout}")
st.write("Instructions: Prepare your camera and click 'Start Workout' when ready.")

if st.button("Start Workout"):
    video_processor = VideoProcessor(workout_plans[selected_workout])
    video_processor.process_video()

    # Display real-time camera feed (for user experience)
    cap = cv2.VideoCapture(0)
    video_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Missing camera frame.")
            break

        # Basic pose estimation (optional, for visual feedback)
        pose_landmarks, annotated_frame = process_frame_for_pose(
            frame, selected_workout
        )
        video_placeholder.image(annotated_frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
