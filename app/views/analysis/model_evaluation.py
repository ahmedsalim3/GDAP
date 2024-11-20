###################################################################################################################################
# Fourth page of the Analysis section:
#   - Model evaluation: visualize model performance on validation or test data
#   - Threshold selection and evaluation plot generation
###################################################################################################################################


import streamlit as st
from streamlit import session_state as _state

from utils.state import check_state, manage_state
from utils import ui as UI

# from utils.style import page_layout

# ==========================
# PAGE LAYOUT AND INTERFACE
# ==========================

# Set the page layout
# page_layout(max_width='100%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Model Evaluation",
    unsafe_allow_html=True,
)

st.write(
    """
    DESCRIPTIVE TEXTS
    
    """
)

st.divider()

# Columns setup
col1, col2 = st.columns([14, 3], gap="small", vertical_alignment="top")
# ~ parameters & evaluation plot | ~ status

# ==========================
# PARAMETERS SECTION
# ==========================

with col1.container(border=True):
    # st.markdown('<h3 style="text-align: center;">Model Evaluation</h3>', unsafe_allow_html=True)

    c1, c2, _ = st.columns([5, 5, 15], vertical_alignment="top")
    evaluate_data = c1.radio("Data to visualize", ["Validation Data", "Test Data"])
    threshold = c2.slider(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        value=0.5,
    )

    evaluate_model = c2.button("Evaluate model")


# ==========================
# EVALUATION PLOT
# ==========================

if evaluate_model:
    if not check_state("classifier"):
        col1.error(
            "No model has been trained yet. Please train a model before proceeding with evaluation"
        )

    if check_state(
        "previous_classifier", "X_test", "y_test", "X_val", "y_val", check_all=True):
        
        if _state["previous_classifier"] == "TensorFlow":
            from models.tesnorfloow import TensorFlowModel as TF

            test_results, val_results = TF.valid(_state, threshold=threshold)
            with col1:
                TF.evaluate(_state, threshold, evaluate_data)

        else:
            from models.sklearn import SkLearn as SK

            test_results, val_results = SK.valid(_state, threshold=threshold)
            with col1:
                SK.evaluate(_state, threshold, evaluate_data)
        
        UI.task_status("Evaluating Model", "✅")
        UI.results_status(
            f"(Test Set)",
            test_results,
            dict_name="model_results",
        )
        UI.results_status(
            f"(Validation Set)",
            val_results,
            dict_name="model_results",
        )


# Show task status
UI.show_task_status(col2, expand=False)

# Model training results
UI.show_model_results(col2)

# DELETE IN PRODUCTION
manage_state(col2)
