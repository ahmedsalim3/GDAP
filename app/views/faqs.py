#######################################################################################################################################
# FAQ page of the BI ML Disease Prediction app. Displays the content from a markdown content file
#######################################################################################################################################

import streamlit as st

from pathlib import Path

st.markdown((Path(__file__).parents[2] / "docs/reports/faqs.md").read_text())
