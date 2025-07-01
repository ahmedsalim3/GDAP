#######################################################################################################################################
# FAQ page of the BI ML Disease Prediction app. Displays the content from a markdown content file
#######################################################################################################################################

from pathlib import Path

import streamlit as st

st.markdown((Path(__file__).parents[2] / "docs/reports/faqs.md").read_text())
