#######################################################################################################################################
# Home page of the app. Displays the app title, an image, and an introductory markdown report.
#######################################################################################################################################

import streamlit as st
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
IMAGE = PROJ_ROOT / "docs/images/app_home.png"

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Machine Learning - Gene-Disease Association Prediction",
    unsafe_allow_html=True,
)

_, center, _ = st.columns([1, 10, 1])
center.image(str(IMAGE))

st.markdown((PROJ_ROOT / "docs/reports/introduction.md").read_text())
