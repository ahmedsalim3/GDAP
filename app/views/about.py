#######################################################################################################################################
# The final page of the app, "About Us". It fetches and displays the markdown content from the About page of the project website.
# The content is retrieved via a raw URL from the GitHub repository and processed using a regular expression to clean up the front matter
#######################################################################################################################################

import re

import requests
import streamlit as st

# https://regex101.com/r/wNRIGH/2

regex = r"(?:^|\n)---[\s\S]*?---\s\n"

url = "https://raw.githubusercontent.com/mentorchains/BI-ML_Disease-Prediction_2024_Site/refs/heads/ahmed/_docs/about.md"
test_str = requests.get(url).text

result = re.sub(regex, "", test_str)
result = result.replace("## ", "### ")

st.markdown(result)
