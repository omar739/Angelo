import streamlit as st
import base64
import torch
import tempfile
from pytorchvideo.models.hub import slowfast_r50
from preprocessing import multi_clip_predict, class_names
import matplotlib.pyplot as plt
# ----------------- PAGE SETUP -----------------

st.set_page_config(page_title="Hello Angelo", page_icon="📷")

def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64("76826.jpg")

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpg;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="background:#327da8;padding:10px;border-radius:6px;">
        Hello Angelo, this is just a demo for you
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h2 style="background:#327da8;padding:10px;border-radius:6px;">
        Upload your video below to see results
    </h2>
    """,
    unsafe_allow_html=True
)

# ----------------- MODEL -----------------

@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = slowfast_r50(pretrained=False)
    model.blocks[-1].proj = torch.nn.Linear(
        model.blocks[-1].proj.in_features, 2
    )
    model.load_state_dict(
        torch.load("slowfast_best_phase2.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----------------- VIDEO UPLOAD -----------------

video_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "mov", "avi"]
)

if video_file is not None:
    st.video(video_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    with st.spinner("Analyzing video..."):
        pred, probs = multi_clip_predict(model, video_path)

    if pred is not None:
        st.success(f"Prediction: **{class_names[pred]}**")

        # Confidence bar graph
        fig, ax = plt.subplots()
        bars = ax.bar(class_names, probs)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        ax.set_title("Model Confidence")

        for bar, p in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                p + 0.02,
                f"{p:.2f}",
                ha="center",
                fontsize=10,
                fontweight="bold"
            )

        st.pyplot(fig)

    else:
        st.error("Could not process video.")
