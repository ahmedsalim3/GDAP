import streamlit as st
from src.edge_predictions import predict, prediction_results
from app.ui import Predictions_intro, display_predicted_data, display_training_status
from app.utils import update_process_tracker, convert_df

# ------------------------
# PREDICTION AND RESULTS
# ------------------------

# side bar
pred_cont = st.sidebar.expander("Model Prediction", True)
with pred_cont:
    data = st.selectbox(
        "Prediction Data:",
        options=["Validation data", "Testing data"],
        help="While the testing set was used by the model during training, the validation data has not been seen by the model",
    )

    threshold = st.number_input(
        "Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        format="%.2f",
    )
    threshold_button = st.button("Make Predictions")


# Introduction
Predictions_intro()

col1, col2 = st.columns([10, 2])
with col1:
    if threshold_button:
        with st.spinner("Processing predictions..."):
            if (
                st.session_state.classifier is None
                or st.session_state.X_val is None
                or st.session_state.edges_val is None
            ):

                st.error(
                    "No model or data available. Please do them on previous pages."
                )
            else:
                if data == "Validation data":
                    classifier = st.session_state["classifier"]
                    X_val = st.session_state["X_val"]
                    edges_val = st.session_state["edges_val"]

                    associated_proteins, non_associated_proteins = predict(
                        classifier, X_val, edges_val, threshold=threshold
                    )
                    associated_df, non_associated_df = prediction_results(
                        associated_proteins, non_associated_proteins
                    )

                    st.session_state.associated_df = associated_df
                    st.session_state.non_associated_df = non_associated_df
                    st.session_state.associated_csv = convert_df(associated_df)
                    st.session_state.non_associated_csv = convert_df(non_associated_df)

                    update_process_tracker("Prediction Completed", "✔️ Completed")

                else:
                    classifier = st.session_state["classifier"]
                    X_test = st.session_state["X_test"]
                    edges_test = st.session_state["edges_test"]

                    associated_proteins, non_associated_proteins = predict(
                        classifier, X_test, edges_test, threshold=threshold
                    )
                    associated_df, non_associated_df = prediction_results(
                        associated_proteins, non_associated_proteins
                    )

                    st.session_state.associated_df = associated_df
                    st.session_state.non_associated_df = non_associated_df
                    st.session_state.associated_csv = convert_df(associated_df)
                    st.session_state.non_associated_csv = convert_df(non_associated_df)

                    update_process_tracker("Prediction Completed", "✔️ Completed")

    display_predicted_data()


with col2:
    display_training_status(False)
