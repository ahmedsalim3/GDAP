import streamlit as st
from app.model_parms import *
from app.model_training import train_classifier, validate_classifier, train_tf
from app.utils import update_process_tracker, update_results, reset_tf_param
import joblib
import io
from app.ui import Model_Training_intro, display_training_status
from app.visualizations import ModelEvaluation


# ------------------------
# MODEL SELECTION SECTION
# ------------------------

# Sidebar for model selection
mod_cont = st.sidebar.expander("Model Selection", True)
with mod_cont:
    st.header("Model Selection and Parameters")
    classifier_options = st.selectbox(
        "Select Classifier:",
        options=["Logistic Regression", "Random Forest", "Gradient Boosting", "SVC", "TensorFlow"],
    )

    # Reset classifier state if the model is changed
    if classifier_options != st.session_state.classifier_name:
        st.session_state["classifier"] = None
        st.session_state["results_tracker"] = {}
        st.session_state["classifier_name"] = classifier_options
        

    if classifier_options == "Logistic Regression":
        classifier = lr_param()
    elif classifier_options == "Random Forest":
        classifier = rf_param()
    elif classifier_options == "Gradient Boosting":
        classifier = gb_param()
    elif classifier_options == "SVC":
        classifier = svc_param()
    elif classifier_options == "TensorFlow":
        classifier = tf_param()
        reset_tf_param(classifier_options, classifier)

    c1, c2 = st.columns([2, 2])
    with c1:
        classifier_buttons = st.button("Train model")

# Sidebar for model evaluation
mod_cont2 = st.sidebar.expander("Model Evaluation", True)
with mod_cont2:
    st.header("Select Data to evaluate")
    c3, c4, c5 = st.columns(3, vertical_alignment="bottom")
    with c3:
        plot_option = st.radio("Data to visualize", ["Validation Data", "Test Data"])
    with c4:
        threshold = st.slider(
            "Threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.5,
        )
    with c5:
        evaluate_model = st.button("Evaluate model")


# ------------------------------
# MODEL TRAINING AND EVALUATION
# ------------------------------

# Introduction
Model_Training_intro()
classifier_error_flag = True
col1, col2 = st.columns([10, 2])

with col1:
    if st.session_state.X_train is not None and st.session_state.y_train is not None:
        X_train = st.session_state["X_train"]
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
        X_val = st.session_state["X_val"]
        y_val = st.session_state["y_val"]

        if st.session_state.classifier is not None:
            classifier = st.session_state["classifier"]
            classifier_error_flag = False
            
        # Train the classifier and get results
        if classifier_buttons:
            reset_tf_param(classifier_options, classifier)
            st.session_state["results_tracker"] = {}
            st.session_state["classifier"] = None
            
            if classifier_options == "TensorFlow":
                import tensorflow as tf
                X_train = tf.constant(X_train)
                y_train = tf.constant(y_train)
                X_test = tf.constant(X_test)
                y_test = tf.constant(y_test)
                X_val = tf.constant(X_val)
                y_val = tf.constant(y_val)
                classifier, st.session_state['history'], acc, loss = train_tf(X_train, y_train, X_test, y_test, classifier)
                st.session_state["classifier"] = classifier
                update_results(
                    f"{classifier_options} Training Performance",
                    f"Test Accuracy {acc:.4f}\nTest Loss: {loss * 2:.4f}" 
                )
                update_process_tracker(
                    f"{classifier_options} Model Trained", "✔️ Completed"
                )
                
            else:  
                classifier, cv_scores, classifier_error_flag = train_classifier(
                    classifier, model_name=classifier_options
                )
                st.session_state["classifier"] = classifier

                if cv_scores is not None:
                    update_results(
                        f"{classifier_options} Cross-Validation:",
                        f"Mean F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})",
                    )
                    update_process_tracker(
                        f"{classifier_options} Model Trained", "✔️ Completed"
                    )

                    st.session_state.classifier_buffer = io.BytesIO()
                    joblib.dump(classifier, st.session_state.classifier_buffer)
                    st.session_state.classifier_buffer.seek(0)

        # Evaluate and plot the model metrics if the button is pressed
        if evaluate_model and st.session_state.classifier is not None:
            if classifier_options == "TensorFlow":
                test_results, val_results, validate_error_flag = validate_classifier(
                    classifier, threshold=threshold, is_tf_model=True
                )
                if plot_option == "Validation Data":
                    Evaluation = ModelEvaluation(
                        classifier,
                        X_val,
                        y_val,
                        threshold=threshold,
                        model_name=classifier_options,
                        figsize=(12, 14),
                        is_tf_model=True,
                        history=st.session_state['history'],
                        history_figsize=(12, 5)
                    )
                    Evaluation.plot_history()
                    Evaluation.plot_evaluation()
                elif plot_option == "Test Data":
                    Evaluation = ModelEvaluation(
                        classifier,
                        X_test,
                        y_test,
                        threshold=threshold,
                        model_name=classifier_options,
                        figsize=(12, 14),
                        is_tf_model=True,
                        history=st.session_state['history'],
                        history_figsize=(12, 5)
                    )
                    Evaluation.plot_history()
                    Evaluation.plot_evaluation()
            else:
                test_results, val_results, validate_error_flag = validate_classifier(
                    classifier, threshold=threshold
                )
                if plot_option == "Validation Data":
                    Evaluation = ModelEvaluation(
                        classifier,
                        X_val,
                        y_val,
                        threshold=threshold,
                        model_name=classifier_options,
                        figsize=(12, 14),
                    )
                    Evaluation.plot_evaluation()
                elif plot_option == "Test Data":
                    Evaluation = ModelEvaluation(
                        classifier,
                        X_test,
                        y_test,
                        threshold=threshold,
                        model_name=classifier_options,
                        figsize=(12, 14),
                    )
                    Evaluation.plot_evaluation()

            update_results(f"{classifier_options} (Test Set)", test_results)
            update_results(f"{classifier_options} (Validation Set)", val_results)
    else:
        if classifier_buttons:
            st.error(
                "No data available for training. Please ensure you have prepared your dataset on the previous page"
            )

    if st.session_state.classifier is None and evaluate_model:
        st.error(
            "No model has been trained yet. Please train a model before proceeding with evaluation"
        )


with col2:
    display_training_status(classifier_error_flag)
