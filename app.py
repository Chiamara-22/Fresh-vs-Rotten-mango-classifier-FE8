import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="🥭 Mango Freshness Classifier",
    page_icon="🥭",
    layout="centered"
)

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#FFF8E1,#E8F5E9);
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#2E7D32;
}

.subtitle{
    text-align:center;
    color:#555;
    font-size:18px;
    margin-bottom:30px;
}

.result-box{
    padding:18px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

div[data-testid="stMetric"]{
    background-color:white;
    padding:15px;
    border-radius:12px;
    border:1px solid #ddd;
    box-shadow:0 2px 8px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    height:50px;
    font-size:18px;
    border-radius:12px;
    background:#2E7D32;
    color:white;
}

.stButton>button:hover{
    background:#1B5E20;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/custom_cnn_best(1).keras")

model = load_model()

# --------------------------
# CLASS NAMES
# --------------------------
class_names = [
    "Fresh Mango",
    "Rotten Mango"
]

# --------------------------
# IMAGE PREPROCESSING
# --------------------------
IMG_SIZE = (160, 160)

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image).astype(np.float32)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image /= 255.0
    image = np.expand_dims(image, axis=0)

    return image

# --------------------------
# HEADER
# --------------------------
st.markdown(
    '<div class="title">🥭 Mango Freshness Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload a mango image and let AI determine whether it is Fresh or Rotten.</div>',
    unsafe_allow_html=True
)

# --------------------------
# FILE UPLOADER
# --------------------------
uploaded_file = st.file_uploader(
    "Upload Mango Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    st.write("")

    if st.button(" Classify Mango"):

        with st.spinner("Analyzing image..."):

            processed = preprocess_image(image)

            prediction = model.predict(processed, verbose=0)[0]

        st.divider()

        # --------------------------------
        # SOFTMAX MODEL (2 OUTPUTS)
        # --------------------------------
        if len(prediction) == 2:

            fresh_prob = float(prediction[0])
            rotten_prob = float(prediction[1])

            predicted_index = np.argmax(prediction)
            predicted_class = class_names[predicted_index]
            confidence = float(prediction[predicted_index])

        # --------------------------------
        # SIGMOID MODEL (1 OUTPUT)
        # --------------------------------
        else:

            rotten_prob = float(prediction[0])
            fresh_prob = 1 - rotten_prob

            if fresh_prob > rotten_prob:
                predicted_class = "Fresh Mango"
                confidence = fresh_prob
            else:
                predicted_class = "Rotten Mango"
                confidence = rotten_prob

        # --------------------------
        # RESULT
        # --------------------------
        if predicted_class == "Fresh Mango":

            st.markdown(
                f"""
                <div class="result-box" style="background:#E8F5E9;color:#1B5E20;">
                ✅ Fresh Mango
                <br><br>
                Confidence: {confidence*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-box" style="background:#FFEBEE;color:#C62828;">
                ⚠️ Rotten Mango
                <br><br>
                Confidence: {confidence*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        # --------------------------
        # PROBABILITIES
        # --------------------------
        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🥭 Fresh",
                f"{fresh_prob*100:.2f}%"
            )

            st.progress(fresh_prob)

        with col2:

            st.metric(
                "🍂 Rotten",
                f"{rotten_prob*100:.2f}%"
            )

            st.progress(rotten_prob)

        st.write("")

        # --------------------------
        # OVERALL CONFIDENCE
        # --------------------------
        st.subheader(" Overall Confidence")

        st.progress(confidence)

        st.metric(
            "Model Confidence",
            f"{confidence*100:.2f}%"
        )

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")

