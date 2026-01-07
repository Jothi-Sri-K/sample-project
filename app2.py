import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
from huggingface_hub import snapshot_download

# --------------------
# Load the exported SavedModel
# --------------------
@st.cache_resource
@st.cache_resource
def load_model():
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

    from huggingface_hub import snapshot_download
    model_path = snapshot_download("sayakpaul/whitebox-cartoonizer")
    model = tf.saved_model.load(model_path)
    return model


model = load_model()

# --------------------
# Preprocessing / postprocessing
# --------------------
def preprocess_image(img: np.ndarray) -> tf.Tensor:
    # Resize / crop logic similar to original repo
    h, w, _ = img.shape
    # Bound max size, maintain aspect ratio (matching Hugging Face example)
    max_dim = 720
    if min(h, w) > max_dim:
        if h > w:
            new_h = int(max_dim * h / w)
            new_w = max_dim
        else:
            new_h = max_dim
            new_w = int(max_dim * w / h)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Make dimensions divisible by 8
    h2 = (img.shape[0] // 8) * 8
    w2 = (img.shape[1] // 8) * 8
    img = img[:h2, :w2, :]
    # Normalize to [-1, 1]
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return tf.constant(img)

def postprocess(tensor: tf.Tensor) -> Image.Image:
    arr = tensor.numpy()[0]  # shape: (H, W, 3)
    arr = (arr + 1.0) * 127.5
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(arr)

# --------------------
# Streamlit UI
# --------------------
st.title("🎨 Whitebox Cartoonizer (HuggingFace version)")

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded:
    image_pil = Image.open(uploaded).convert("RGB")
    st.subheader("Original")
    st.image(image_pil, use_container_width=True)

    # Convert PIL to numpy
    img_np = np.array(image_pil)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Preprocess
    input_tensor = preprocess_image(img_np)

    # Run inference
    out = model.signatures["serving_default"](input_tensor)["final_output:0"]

    # Postprocess
    cartoon_img = postprocess(out)

    st.subheader("Cartoonized")
    st.image(cartoon_img, use_container_width=True)

    # Download button
    buf = io.BytesIO()
    cartoon_img.save(buf, format="PNG")
    st.download_button(
        "Download Cartoonized Image",
        data=buf.getvalue(),
        file_name="cartoonized.png",
        mime="image/png"
    )

#using whitebox huggingface model
