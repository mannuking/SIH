import streamlit as st
# --- Page Configuration ---
st.set_page_config(page_title="WorkoutAI", page_icon=":muscle:") # <- Moved to the top!

import cv2
from utils.pose_estimation import process_frame_for_pose
from utils.llava_interaction import get_feedback_from_llava, load_llava_model
from video_processor import VideoProcessor
from workout_plans import get_workout_plans

# Initialize session state variables
if "workout_active" not in st.session_state:
    st.session_state.workout_active = False
if "video_processor" not in st.session_state:
    st.session_state.video_processor = None

# --- Page Title and Description ---
st.title("WorkoutAI")


# Load LLaVA model and display status
model, processor = load_llava_model()
if model is not None and processor is not None:
    st.success("LLaVA model loaded successfully!")
else:
    st.error("Error loading LLaVA model. Please check your configuration.")

st.write(
    "Instructions: Prepare your camera and click 'Start Workout' when ready. The AI will automatically recognize and analyze your workout."
)

# Get available workout plans
workout_plans = get_workout_plans()

# --- Workout Controls (Top) ---
col1, col2, col3 = st.columns([1, 1, 1])  # Create 3 equal columns
with col2:  # Center the button
    if st.button("Start Workout", key="start_workout_top"):
        st.session_state.workout_active = True
        st.session_state.video_processor = VideoProcessor()

# --- Real-time Workout Display ---
if st.session_state.workout_active:
    cap = cv2.VideoCapture(0)
    video_placeholder = st.empty()
    feedback_placeholder = st.empty()

    while st.session_state.workout_active:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Missing camera frame.")
            break

        # Process frame and get feedback
        pose_landmarks, annotated_frame = process_frame_for_pose(frame)
        if st.session_state.video_processor: 
            st.session_state.video_processor.process_workout(pose_landmarks)

        # Display annotated frame
        video_placeholder.image(annotated_frame, channels="BGR")

        # Add a slight delay
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()

# --- Workout Controls (Bottom) ---
if st.session_state.workout_active:
    if st.button("Stop Workout", key="stop_workout_bottom"):
        st.session_state.workout_active = False

# --- AI Feedback and Summary (After Workout) ---
if not st.session_state.workout_active and st.session_state.video_processor:
    st.write("Workout Summary:")
    if st.session_state.video_processor.current_workout:
        st.write(
            f"Workout type: {st.session_state.video_processor.current_workout}"
        )
        st.write(
            f"Total sets completed: {st.session_state.video_processor.set_count}"
        )
        st.write(
            f"Total reps completed: {st.session_state.video_processor.rep_count}"
        )

        # --- Display AI Feedback ---
        if st.session_state.video_processor.ai_feedback:
            st.write("## AI Trainer Feedback:")
            st.write(st.session_state.video_processor.ai_feedback)
            # st.session_state.video_processor.provide_audio_feedback(
            #     st.session_state.video_processor.ai_feedback
            # )

    # --- Reset the video processor ---
    st.session_state.video_processor = None

