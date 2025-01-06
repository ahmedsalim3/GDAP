###################################################################################################################################
# Final page of the Analysis section:
#   - Model prediction: make predictions on validation or testing data
#   - Display results for associated and non-associated genes
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state

from utils.state import persist, check_state, manage_state
from utils import ui as UI

# from utils.style import page_layout


# ==========================
# PAGE LAYOUT AND INTERFACE
# ==========================

# Set the page layout
# page_layout(max_width='100%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Model Prediction",
    unsafe_allow_html=True,
)

st.write(
    """
    DESCRIPTIVE TEXTS
    
    """
)

st.divider()

# Columns setup
col1, col2, col3 = st.columns([3.5, 5, 2], gap="small", vertical_alignment="top")

# ==========================
# PARAMETERS SECTION
# ==========================

with col1.container(border=True):

    st.markdown("##### Model Prediction")

    left, right = st.columns(2, vertical_alignment="top")

    data = left.selectbox(
        "Prediction Data:",
        options=["Validation data", "Testing data"],
        help="While the testing set was used by the model during training, the validation data has not been seen by the model",
    )

    threshold = left.number_input(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.2f",
    )

    right.markdown("<br><br><br>", unsafe_allow_html=True)
    prediction_button = right.button("Make Predictions")


# ==========================
# MAKE PREDICTIONS
# ==========================

if prediction_button:
    if not check_state("classifier"):
        col1.error(
            "No model has been trained yet. Please train a model before proceeding with predictions"
        )

    if check_state(
        "classifier", "X_val", "X_test", "edges_val", "edges_test", check_all=True
    ):

        if data == "Validation data":
            model = _state.classifier
            X_val = _state.X_val
            edges_val = _state.edges_val

            from gene_disease.edges.edge_predictions import predict, prediction_results

            associated_proteins, non_associated_proteins = predict(
                model, X_val, edges_val, threshold=threshold
            )
            associated_df, non_associated_df = prediction_results(
                associated_proteins, non_associated_proteins
            )

            persist("associated_df", associated_df)
            persist("non_associated_df", non_associated_df)

            UI.task_status("Validation Prediction Completed", "✅")
            UI.flash_message(
                "associated_df",
                "non_associated_df",
                message=f"predictions for {_state['disease_name']} were made successfully!\n\nValidation Length: {len(X_val)}",
                col=col1,
            )
            
            st.balloons()
            st.snow()
        
        elif data == "Testing data":
            model = _state.classifier
            X_test = _state.X_test
            edges_test = _state.edges_test

            from gene_disease.edges.edge_predictions import predict, prediction_results

            associated_proteins, non_associated_proteins = predict(
                model, X_test, edges_test, threshold=threshold
            )
            associated_df, non_associated_df = prediction_results(
                associated_proteins, non_associated_proteins
            )

            persist("associated_df", associated_df)
            persist("non_associated_df", non_associated_df)

            UI.task_status("Testing Prediction Completed", "✅")
            UI.flash_message(
                "associated_df",
                "non_associated_df",
                message=f"predictions for {_state['disease_name']} were made successfully!\n\nTesting Length: {len(X_test)}",
                col=col1,
            )
            
            st.balloons()
            st.snow()

# ==========================
# DISPLAYING RESULTS
# ==========================

if check_state("associated_df", "non_associated_df"):
    tab1, tab2 = col2.tabs(["🗃 associated genes", "🗃 non-associated genes"])

    expander = tab1.expander(
        f"Click here to see associated genes to {_state['disease_name']}", expanded=True
    )
    expander.dataframe(_state["associated_df"], width=1000)

    from streamlit_helpers.cache_data import convert_df

    expander.download_button(
        label=f"Download associated predection for {_state['disease_name']} disease",
        data=convert_df(_state["associated_df"]),
        file_name=f"{_state['disease_name']}_associated_predection.csv",
        mime="text/csv",
    )

    expander2 = tab2.expander(
        f"Click here to see non-associated genes to {_state['disease_name']}",
        expanded=True,
    )
    expander2.dataframe(_state["non_associated_df"], width=1000)

    expander2.download_button(
        label=f"Download non-associated predection for {_state['disease_name']} disease",
        data=convert_df(_state["non_associated_df"]),
        file_name=f"{_state['disease_name']}_non_associated_predection.csv",
        mime="text/csv",
    )

# Show task status
UI.show_task_status(col3, expand=False)

# Show model training results
UI.show_model_results(col3)

# DELETE IN PRODUCTION
manage_state(col3)
