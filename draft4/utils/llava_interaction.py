import torch
import cv2
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import streamlit as st
import tensorflow as tf
import tempfile
import av

# Force TensorFlow to use GPU if available 
with tf.device('/GPU:0'):
    print('GPU is available') 

# Load LLaVA model and processor once, and cache the result
@st.cache_resource
def load_llava_model():
    """Loads the LLaVA model and processor with caching."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"  
        model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(device)
        processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        return model, processor
    except Exception as e:
        print(f"Error loading LLaVA model: {e}")
        return None, None  # Return None if there's an error

model, processor = load_llava_model()



def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def analyze_video_clip(video_path, exercise_type):
    """Analyzes a 15-second video clip and provides feedback."""
    if model is None or processor is None:
        return "Error: LLaVA model not loaded."

   
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)   #SKIPPING EVERY 8TH FRAME
    set_clip = read_video_pyav(container, indices)
    # cap = cv2.VideoCapture(video_path)
    # frames = []
    # frame_count = 0
    # while frame_count < 15 * 30:  # Assuming 30 fps, process 15 seconds
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     if frame_count % 5 == 0:  # Sample every 10th frame to reduce processing
    #         resized_frame = cv2.resize(frame, (224, 224))  # Resize frames to match LLaVA input
    #         frames.append(resized_frame)
    #     frame_count += 1
    # cap.release()
    

    if not set_clip:
        return "No frames found in the video clip."

    # frames = np.array(frames) 

    # Construct the conversation prompt for LLaVA
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Analyze this 15-second {exercise_type} exercise video clip. Provide detailed feedback on the form, including specific observations and motivational comments. Mention any improvements needed and positive aspects of the performance. Focus on the overall quality of the exercise execution.",
                },
                {"type": "video"}, # Indicate that video input follows
            ],
        },
    ]

    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(
        text=prompt, videos=set_clip, padding=True, return_tensors="pt"
    ).to(model.device) 

    # Generate response from LLaVA
    output = model.generate(**inputs_video, max_new_tokens=200, do_sample=False)
    response = processor.batch_decode(output, skip_special_tokens=True)

    return response

def get_feedback_from_llava(video_path, exercise_type):
    """Gets feedback from LLaVA based on a 15-second video clip."""
    feedback = analyze_video_clip(video_path, exercise_type)
    return feedback
