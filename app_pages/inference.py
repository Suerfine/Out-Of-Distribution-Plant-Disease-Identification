import streamlit as st

st.info(
    "This page will take a plant image, predict its crop–disease pair, and show the most similar training "
    "images. It is enabled once a trained model checkpoint is available.",
    title="No model yet",
    icon=":material/model_training:",
)

upload = st.file_uploader("Plant image", type=["jpg", "jpeg", "png"], disabled=True)
