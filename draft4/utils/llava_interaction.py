import torch
import cv2
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
import streamlit as st


# Load LLaVA model (only once)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "llava-hf/LLaVA-NeXT-Video-7B-hf"


@st.cache_resource
def load_llava_model():
    """Loads the LLaVA model and processor with caching."""
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to(device)
    processor = LlavaNextVideoProcessor.from_pretrained(model_id)
    return model, processor


model, processor = load_llava_model()


def get_feedback_from_llava(frame, exercise_type):
    """Gets feedback from LLaVA based on the frame and exercise type."""

    # Prepare input for LLaVA (resize frame, create conversation)
    resized_frame = cv2.resize(frame, (224, 224))
    resized_frame = resized_frame.astype(np.uint8)

    if np.all(resized_frame == 0):
        st.warning("Skipping empty frame.")
        return "No feedback available (empty frame)."

    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Analyze this {exercise_type} exercise and provide feedback on the form.",
                },
                {"type": "video"},
            ],
        },
    ]

    # Process input and generate response
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs_video = processor(
        text=prompt, videos=resized_frame, padding=True, return_tensors="pt"
    ).to(device)
    output = model.generate(**inputs_video, max_new_tokens=100, do_sample=False)
    response = processor.decode(output[0][2:], skip_special_tokens=True)

    return response
