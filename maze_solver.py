import streamlit as st
import base64


def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
bg_image = get_base64("winter.jpg")

page_bg = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.set_page_config(page_title="Angelo",
                   page_icon="📷",)


st.markdown("""
<style>
.highlight {
    background-color: #327da8;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-weight: 600;
}
</style>

<h1><span class="highlight">Hello Angelo, this is just a demo for you</span></h1>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.highlight {
    background-color: #327da8;
    padding: 0.2em 0.4em;
    border-radius: 4px;
    font-weight: 600;
}
</style>

<h2><span class="highlight">Upload your video below to see results</span></h2>
""", unsafe_allow_html=True)

