###################################################################################################################################
#  Utility functions to save a classifier model to memory and provide a download button for exporting the model file.
#  NOTE: TensorFlow models are not supported for download.  
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
import io
import joblib
    
def download_clasifer(classifier):
    _state.classifier_buffer = io.BytesIO()
    joblib.dump(classifier, _state.classifier_buffer)
    _state.classifier_buffer.seek(0)
    
def download_classifier_button():
    if _state.classifier_options == "TensorFlow":
        return

    st.download_button(
        label=f"Download {_state.classifier_options} model",
        data=_state.classifier_buffer,
        file_name=f"{_state.classifier_options}_{_state.disease_name}.pkl",
        mime="application/octet-stream",
    )