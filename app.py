# app.py
import streamlit as st

# Set page config as the very first Streamlit command
st.set_page_config(page_title="StoryLens - AI Photo Story Generator", layout="centered")

from PIL import Image
import requests
import torch
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers.utils.hub import cached_file
from TTS.api import TTS
import tempfile
import uuid
import os
import time
import traceback

# Debug mode toggle
debug_mode = False

# Load Kosmos-2 with error handling
@st.cache_resource
def load_kosmos():
    try:
        st.info("Loading Kosmos-2 model...")
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        model = AutoModelForImageTextToText.from_pretrained("microsoft/kosmos-2-patch14-224")
        # Ensure model is on CPU to avoid CUDA issues
        if hasattr(model, 'to'):
            model = model.to('cpu')
        st.success("Model loaded successfully!")
        return processor, model
    except (requests.exceptions.ConnectionError, OSError) as e:
        st.error(f"Network error when loading model: {str(e)}")
        st.warning("Please check your internet connection and try again later.")
        st.info("Alternatively, try restarting the application.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        if debug_mode:
            st.error(f"Stack trace: {traceback.format_exc()}")
        return None, None

# Load TTS with error handling
@st.cache_resource
def load_tts():
    try:
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    except Exception as e:
        st.warning(f"Failed to load TTS model: {str(e)}")
        st.info("Story will be generated without audio narration.")
        if debug_mode:
            st.error(f"Stack trace: {traceback.format_exc()}")
        return None

# Debug sidebar
with st.sidebar:
    st.title("StoryLens Settings")
    debug_mode = st.checkbox("Debug Mode", value=False)
    if debug_mode:
        st.write("Python version:", sys.version)
        st.write("PyTorch version:", torch.__version__ if 'torch' in sys.modules else "Not loaded")
        st.write("Device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Try to load the models with proper error handling
processor, model = load_kosmos()

# Check if models loaded successfully
if None in (processor, model):
    st.error("Failed to load Kosmos-2 model. Application may not work correctly.")
    st.stop()

# Load TTS model
tts = load_tts()

st.title("📷 StoryLens")
st.markdown("Upload an image and get a story or poem narrated in an AI-generated voice!")

uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🖋️ Generate Story & Narration"):
        with st.spinner("Generating story..."):
            try:
                # Generate story with additional error checking
                if processor is None or model is None:
                    raise ValueError("Model or processor is not properly loaded")
                
                # Process the image
                if debug_mode:
                    st.info("Processing image...")
                
                inputs = processor(images=image, return_tensors="pt")
                
                if debug_mode:
                    st.info(f"Input keys: {list(inputs.keys())}")
                    for k, v in inputs.items():
                        if hasattr(v, 'shape'):
                            st.info(f"{k} shape: {v.shape}")
                
                # Verify inputs are valid
                if not inputs:
                    raise ValueError("Failed to process image - empty inputs")
                
                # Check for required keys based on model type
                required_keys = ['pixel_values']  # Minimum required
                missing_keys = [key for key in required_keys if key not in inputs]
                if missing_keys:
                    raise ValueError(f"Missing required inputs: {missing_keys}")
                
                # Ensure model is on CPU
                if hasattr(model, 'to'):
                    model = model.to('cpu')
                
                if debug_mode:
                    st.info("Moving tensors to CPU...")
                
                # Move tensors to CPU safely
                processed_inputs = {}
                for k, v in inputs.items():
                    if v is not None:
                        if isinstance(v, torch.Tensor):
                            processed_inputs[k] = v.to('cpu')
                        else:
                            processed_inputs[k] = v
                    else:
                        # Skip None values
                        if debug_mode:
                            st.warning(f"Skipping None value for key {k}")
                
                if debug_mode:
                    st.info("Generating text...")
                
                # Generate text
                generated_ids = model.generate(**processed_inputs, max_new_tokens=100)
                
                # Verify generation succeeded
                if generated_ids is None:
                    raise ValueError("Model failed to generate text")
                
                if debug_mode:
                    st.info(f"Generated IDs shape: {generated_ids.shape}")
                    
                # Decode the text
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)
                if not caption or len(caption) == 0:
                    raise ValueError("Failed to decode generated text")
                
                story = f"Once upon a time, {caption[0].strip('.')} in a magical moment."
                
                st.markdown("### 📖 Generated Story")
                st.success(story)
                
                # Generate audio if TTS is available
                if tts is not None:
                    with st.spinner("Generating voice..."):
                        tmp_path = os.path.join(tempfile.gettempdir(), f"storylens_{uuid.uuid4().hex}.wav")
                        try:
                            tts.tts_to_file(
                                text=story,
                                speaker_wav=None,
                                language="en",
                                file_path=tmp_path
                            )
                            with open(tmp_path, "rb") as f:
                                st.audio(f.read(), format="audio/wav")
                        except Exception as e:
                            st.error(f"Error generating audio: {str(e)}")
                            if debug_mode:
                                st.error(f"Stack trace: {traceback.format_exc()}")
                        finally:
                            if os.path.exists(tmp_path):
                                try:
                                    os.remove(tmp_path)
                                except:
                                    pass
                else:
                    st.warning("Audio narration is not available due to TTS model loading error.")
                    
            except ValueError as e:
                st.error(f"Error: {str(e)}")
                if debug_mode:
                    st.error(f"Stack trace: {traceback.format_exc()}")
                st.info("Try uploading a different image or restart the application.")
            except Exception as e:
                st.error(f"Error generating story: {str(e)}")
                if debug_mode:
                    st.error(f"Stack trace: {traceback.format_exc()}")
                st.info("Try uploading a different image or restart the application.")
