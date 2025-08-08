"""
Streamlit interface for digit drawing + TrOCR API.
"""

import cv2
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import requests
from PIL import Image
import io

BACKEND_URL = "http://127.0.0.1:8000/predict"
CANVAS_SIZE = 250

def classify_drawn(img):
    """
    Sends the canvas image to backend (POST /predict)
    and returns predicted number string.
    """
    # Convert numpy -> PIL -> Bytes
    im = Image.fromarray(img)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("img.png", buf, "image/png")}
    r = requests.post(BACKEND_URL, files=files)
    out = r.json()
    return out["prediction"]

def main():
    st.set_page_config(page_title="Handwritten Number Recognition")
    st.title("Draw a Number (multiple digits allowed)")
    st.caption("Powered by HuggingFace TrOCR backend (FastAPI)")
    
    col1, col2 = st.columns(2)
    prediction = None

    with col1:
        st.subheader("✏ Draw Here:")
        canvas = st_canvas(
            stroke_color="black",
            background_color="white",
            stroke_width=15,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas.image_data is not None:
            if st.button("Predict"):
                img = cv2.cvtColor(canvas.image_data.astype('uint8'), cv2.COLOR_RGBA2RGB)
                st.spinner("Predicting...")
                prediction = classify_drawn(img)

    if prediction is not None:
        with col2:
            st.subheader("✅ Prediction")
            st.success(f"The number you drew is: *{prediction}*")

if __name__ == "__main__":
    main()