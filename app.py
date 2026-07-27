"""
Fresh vs Rotten Mango Classifier — Streamlit App
Loads the trained custom CNN (models/mango_cnn.keras) and classifies
uploaded mango images as Fresh or Rotten.

Run locally:
    streamlit run app.py

Deploy:
    Push this file + models/mango_cnn.keras + requirements.txt to a repo,
    then point Streamlit Community Cloud (or any host) at app.py.
"""

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ── Config — MUST match what the model was trained on ───────────────────────
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 160
CLASS_NAMES = ["Fresh", "Rotten"]  # index 0 / index 1 — matches mango_class_names order
MODEL_PATH = "models/custom_cnn_best(1).keras"

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mango Freshness Classifier",
    page_icon="🥭",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .main {
        background: linear-gradient(180deg, #fffaf3 0%, #fff3e0 100%);
    }

    .hero {
        text-align: center;
        padding: 1.6rem 1rem 1.2rem 1rem;
        background: linear-gradient(135deg, #ef6c00 0%, #ffb74d 100%);
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(239, 108, 0, 0.25);
    }
    .hero h1 {
        color: white;
        font-weight: 700;
        font-size: 2.1rem;
        margin-bottom: 0.2rem;
    }
    .hero p {
        color: #fff3e0;
        font-size: 1.0rem;
        margin: 0;
    }

    .upload-card {
        background: white;
        border-radius: 16px;
        padding: 1.4rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.06);
        margin-bottom: 1.4rem;
        border: 1px solid #f5ece0;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.6rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .result-fresh {
        background: linear-gradient(135deg, #e8f8ec 0%, #d4f2dc 100%);
        border: 2px solid #4caf50;
    }
    .result-rotten {
        background: linear-gradient(135deg, #fdeceb 0%, #fbdbd8 100%);
        border: 2px solid #e53935;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .result-fresh .result-label { color: #2e7d32; }
    .result-rotten .result-label { color: #c62828; }

    .confidence-text {
        font-size: 1.0rem;
        color: #444;
        margin-top: 0.3rem;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Model loading (cached so it only loads once per session) ────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Resize + array-ify. No manual rescale — the model has its own
    Rescaling(1/255) layer baked in. Rescaling here too would silently
    double-scale the input and wreck predictions."""
    img = pil_image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    arr = tf.keras.utils.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr


def predict(model, pil_image: Image.Image):
    arr = preprocess_image(pil_image)
    prob_rotten = float(model.predict(arr, verbose=0)[0][0])  # sigmoid output
    pred_idx = int(prob_rotten >= 0.5)
    label = CLASS_NAMES[pred_idx]
    confidence = prob_rotten if pred_idx == 1 else 1 - prob_rotten
    return label, confidence, prob_rotten


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🥭 About")
    st.write(
        "This app uses a custom Convolutional Neural Network (CNN) "
        "trained from scratch to classify mangoes as **Fresh** or **Rotten** "
        "from a single image."
    )
    st.markdown("---")
    st.markdown("### ⚙️ Model details")
    st.write(f"- Input size: {IMAGE_WIDTH}×{IMAGE_HEIGHT}")
    st.write("- Architecture: 3-block custom CNN")
    st.markdown("---")


# ── Hero header ───────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🥭 Mango Freshness Classifier</h1>
        <p>Upload a photo of a mango and let the CNN judge its freshness</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Upload card ───────────────────────────────────────────────────────────
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a mango image",
    type=["jpg", "jpeg", "png"],
    help="JPG or PNG. Best results with a clear, well-lit, single-mango photo.",
)
st.markdown("</div>", unsafe_allow_html=True)

# ── Prediction flow ───────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    try:
        model = load_model()
    except Exception as e:
        st.error(
            f"Couldn't load the model from `{MODEL_PATH}`. "
            f"Make sure the file exists in your deployment and matches this path.\n\n"
            f"Details: {e}"
        )
        st.stop()

    with st.spinner("Analysing mango..."):
        label, confidence, prob_rotten = predict(model, image)

    with col2:
        card_class = "result-fresh" if label == "Fresh" else "result-rotten"
        emoji = "✅" if label == "Fresh" else "⚠️"
        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">{emoji} {label}</div>
                <div class="confidence-text">Confidence: {confidence*100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")
        st.progress(confidence)
        st.caption(f"Raw model output (P[Rotten]) = {prob_rotten:.4f}")

else:
    st.info("👆 Upload a mango image to get a prediction.")

st.markdown("---")
st.caption("Built with TensorFlow + Streamlit · Custom CNN, trained from scratch")
