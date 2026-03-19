
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import numpy as np
import cv2
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MRI Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# ─────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    .title { font-size: 2rem; font-weight: 700; color: #1a1a2e; text-align: center; }
    .subtitle { font-size: 1rem; color: #555; text-align: center; margin-bottom: 2rem; }
    .result-box { padding: 1.2rem; border-radius: 10px; text-align: center;
                  font-size: 1.3rem; font-weight: 600; margin-top: 1rem; }
    .tumor    { background-color: #ffe0e0; color: #c0392b; border: 1px solid #e74c3c; }
    .no-tumor { background-color: #e0f7e9; color: #1e8449; border: 1px solid #27ae60; }
    .disclaimer { font-size: 0.8rem; color: #888; text-align: center; margin-top: 2rem; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="title">🧠 MRI Brain Tumor Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an MRI scan to detect brain tumors using AI</div>',
    unsafe_allow_html=True
)
st.divider()

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
MODEL_PATH  = "tumor_model.pth"
NUM_CLASSES = 4
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
# Class mapping (alphabetical — must match training):
#   0 → Glioma  |  1 → Meningioma  |  2 → No Tumor  |  3 → Pituitary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        st.success("✅ Model loaded successfully!")
    else:
        st.warning(
            f"'{MODEL_PATH}' not found. "
            "Place tumor_model.pth in the same folder as app.py."
        )
    model.eval()
    model.to(device)
    return model

model = load_model()

# ─────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────
def generate_gradcam(img_pil):
    gradients  = []
    activations = []

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def save_activation(module, input, output):
        activations.append(output)

    target_layer = model.layer4[-1]
    handle_f = target_layer.register_forward_hook(save_activation)
    handle_b = target_layer.register_backward_hook(save_gradient)

    try:
        original = np.array(img_pil.resize((224, 224)))
        if original.ndim == 2:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)

        input_tensor = transform(img_pil).unsqueeze(0).to(device)

        model.eval()
        output = model(input_tensor)
        pred   = output.argmax()

        model.zero_grad()
        output[0, pred].backward()

        if len(gradients) == 0 or len(activations) == 0:
            return original, pred.item()

        grad    = gradients[0].cpu().data.numpy()[0]
        act     = activations[0].cpu().data.numpy()[0]
        weights = np.mean(grad, axis=(1, 2))

        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)
        cam = cv2.resize(cam, (224, 224))

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

        return overlay, pred.item()

    finally:
        handle_f.remove()
        handle_b.remove()

# ─────────────────────────────────────────────
# Explanation
# ─────────────────────────────────────────────
def generate_explanation(pred):
    explanations = {
        0: ("⚠️ The model detected patterns associated with **Glioma** — "
            "a type of tumor that forms in the brain or spinal cord. "
            "Please consult a medical professional immediately."),
        1: ("⚠️ The model detected patterns associated with **Meningioma** — "
            "a tumor that arises from the membranes surrounding the brain and spinal cord. "
            "Please consult a medical professional."),
        2: ("✅ The model found **no strong indicators of a brain tumor**. "
            "The scan appears within normal range based on the training data. "
            "Always verify results with a qualified doctor."),
        3: ("⚠️ The model detected patterns associated with a **Pituitary tumor** — "
            "a tumor in the pituitary gland at the base of the brain. "
            "Please consult a medical professional."),
    }
    return explanations.get(pred, "Prediction result not recognized.")

# ─────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────
def predict(img_pil):
    model.eval()
    with torch.no_grad():
        tensor = transform(img_pil).unsqueeze(0).to(device)
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = output.argmax().item()

    overlay, _ = generate_gradcam(img_pil)
    explanation = generate_explanation(pred)

    return pred, probs, overlay, explanation

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses a **ResNet50** model fine-tuned on MRI brain scan images "
        "to classify brain tumors into 4 categories."
    )
    st.divider()
    st.write("**Classes detected:**")
    st.write("🔴 Glioma")
    st.write("🔴 Meningioma")
    st.write("🟢 No Tumor")
    st.write("🔴 Pituitary")
    st.divider()
    st.write("**Model:** ResNet50 (transfer learning)")
    st.write("**XAI:** Grad-CAM heatmap")
    st.write("**Device:**", str(device).upper())
    st.divider()
    st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")

# ─────────────────────────────────────────────
# Main — Upload & Predict
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload an MRI scan image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image to analyse"
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded MRI")
        st.image(img, use_column_width=True)

    with st.spinner("Analysing MRI scan..."):
        try:
            pred, probs, overlay, explanation = predict(img)

            with col2:
                st.subheader("Grad-CAM Heatmap")
                st.image(overlay, use_column_width=True)
                st.caption("Red = model focused here most | Blue = less relevant")

            # Result box
            label     = CLASS_NAMES[pred]
            css_class = "no-tumor" if pred == 2 else "tumor"
            st.markdown(
                f'<div class="result-box {css_class}">Prediction: {label}</div>',
                unsafe_allow_html=True
            )

            # Confidence scores
            st.subheader("Confidence Scores")
            for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
                st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

            # Explanation
            st.subheader("Explanation")
            st.info(explanation)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Make sure tumor_model.pth is in the same folder as app.py.")

# ─────────────────────────────────────────────
# Disclaimer
# ─────────────────────────────────────────────
st.markdown(
    '<div class="disclaimer">'
    '⚠️ This tool is for educational and research purposes only. '
    'It is not a substitute for professional medical advice, diagnosis, or treatment.'
    '</div>',
    unsafe_allow_html=True
)
