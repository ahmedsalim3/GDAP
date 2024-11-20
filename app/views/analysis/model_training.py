###################################################################################################################################
# Third page of the Analysis section: 
#  - Embedding generation and feature extraction
#  - Data splitting for training, validation, and testing
#  - Classifier model training and validation
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
from streamlit_helpers.embeddings import embedding_generator
from gene_disease.edges.edge_utils import (
    features_labels_edges,
    map_nodes,
    features_labels_edges_idx,
    split_edge_data,
)

from models import model_parms as MODEL

from utils.models_files import download_clasifer, download_classifier_button
from utils.state import persist, init_values, check_state, manage_state
from utils import ui as UI

# from utils.style import page_layout

# ==========================
# PAGE LAYOUT AND INTERFACE
# ==========================

# Set the page layout
# page_layout(max_width='100%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Embeddings & Model Selection",
    unsafe_allow_html=True,
)

st.write(
    """
    DESCRIPTIVE TEXTS
    
    -------
    """
)

# Columns setup
col1, col2, col3 = st.columns([3, 5, 3], gap="small", vertical_alignment="top")
# ~ embedding model & data splitting | ~ model training | ~ status & results

# ==========================
# PARAMETERS SECTION
# ==========================

with col1.container(border=True):
    st.markdown(
        '<h3 style="text-align: center;">Feature Engineering</h3>',
        unsafe_allow_html=True,
    )
    # st.divider()

    # ================ EMBEDDING SELECTION ================

    st.markdown("##### Embedding Modes")
    col1_left, _ = st.columns([2, 1], vertical_alignment="top")
    embeddings_mode = col1_left.radio(
        "Select Embedding Mode:",
        ["Simple Node Embedding", "Node2Vec Model", "ProNE Model", "GGVec Model"],
        help="While Simple Node Embedding is the fastest option, it may not "
        "provide the best results. We recommend trying either ProNE or "
        "GGVec for improved performance.",
    )

    embedding_button = col1_left.button("Generate Embedding")

    st.markdown("-----")

    # ================ DATA SPLITTING ================

    st.markdown("##### Data Splitting")
    col1_left_buttom, col1_right_bottom = st.columns([2, 1.5], vertical_alignment="top")

    test_size = col1_left_buttom.number_input(
        "Test Split Ratio:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        format="%.2f",  # format to two decimal places
        help="Split ratio for training, validation, and testing. "
        "By default, the test and validation data will be half split.",
    )

    col1_right_bottom.markdown("<br>", unsafe_allow_html=True)
    test_size_button = col1_right_bottom.button("Test Size")

# ================ CLASSIFIER MODEL SELECTION ================

with col2.container(border=True):
    st.markdown(
        '<h3 style="text-align: center;">Model Training</h3>', unsafe_allow_html=True
    )
    init_values("previous_classifier")

    col2_left, right = st.columns([1, 1], vertical_alignment="top")

    classifier_options = col2_left.selectbox(
        "Select Classifier:",
        options=[
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "SVC",
            "TensorFlow",
        ],
    )

    classifier_button = col2_left.button("Train model")

    with right.expander("Model Parameters:", True):
        if classifier_options == "Logistic Regression":
            parms = MODEL.lr_param()
        elif classifier_options == "Random Forest":
            parms = MODEL.rf_param()
        elif classifier_options == "Gradient Boosting":
            parms = MODEL.gb_param()
        elif classifier_options == "SVC":
            parms = MODEL.svc_param()
        elif classifier_options == "TensorFlow":
            parms = MODEL.tf_param()

        persist("parms", parms)
        persist("classifier_options", classifier_options)


# ==========================
# GENERATE EMBEDDINGS
# ==========================

if embedding_button:
    if check_state("G"):
        G = _state["G"]
        pos_edges = _state["pos_edges"]
        neg_edges = _state["neg_edges"]

        with col1_left:
            with st.spinner(f"Generating embeddings using {embeddings_mode}"):
                embeddings = embedding_generator(G, embeddings_mode=embeddings_mode)

        persist("embeddings_mode", embeddings_mode)
        persist("embeddings", embeddings)
        UI.task_status("Generating Embeddings", "✅")
        UI.results_status(f"Embeddings length", len(embeddings))
        UI.flash_message(
            "embeddings",
            message=f"{len(embeddings)} embeddings Generated using {embeddings_mode}!",
            col=col1,
        )

        # ================ FEATURE ENGINEERING ================

        with col1_left:
            with st.spinner(f"Feature extracting"):
                if embeddings_mode == "Simple Node Embedding":
                    X, y, edges = features_labels_edges(
                        pos_edges, neg_edges, embeddings
                    )

                else:
                    node_to_index = map_nodes(G)
                    X, y, edges = features_labels_edges_idx(
                        pos_edges, neg_edges, embeddings, node_to_index
                    )

        persist("X", X)
        persist("y", y)
        persist("edges", edges)
        UI.flash_message(
            "X",
            "y",
            message=f"Feature generated, with a length of {len(X), len(y)}",
            col=col1,
        )

    else:
        col1.error("Please ensure that the graph is available.")


# ==========================
# DATA SPLITTING SECTION
# ==========================

if test_size_button:
    if check_state("X", "y", check_all=True):
        X = _state["X"]
        y = _state["y"]
        edges = _state["edges"]

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            edges_train,
            edges_val,
            edges_test,
        ) = split_edge_data(X, y, edges, test_size=test_size)

        persist("X_train", X_train)
        persist("y_train", y_train)
        persist("X_val", X_val)
        persist("y_val", y_val)
        persist("X_test", X_test)
        persist("y_test", y_test)
        persist("edges_train", edges_train)
        persist("edges_val", edges_val)
        persist("edges_test", edges_test)

        UI.task_status("Splitting data", "✅")
        UI.flash_message(
            "X_train", "y_train", message=f"Data split successfully!", col=col1
        )

        UI.results_status(
            f"Train size", (_state["X_train"].shape, _state["y_train"].shape)
        )
        UI.results_status(
            f"Test size", (_state["X_test"].shape, _state["y_test"].shape)
        )
        UI.results_status(f"Val size", (_state["X_val"].shape, _state["y_val"].shape))

    else:
        col1.error(
            "Cannot split data because features are not defined, please generate embeddings first."
        )


# ==========================
# MODEL TRAINING SECTION
# ==========================

if classifier_button:
    # Reset classifier state if the model has changed
    if _state.previous_classifier != classifier_options:
        persist("classifier", None)
        persist("previous_classifier", classifier_options)

        # Delete previous results from the reesults status
        if check_state("results_status"):
            if "Test accuracy" in _state["results_status"]:
                del _state["results_status"]["Test accuracy"]

            if "Test loss" in _state["results_status"]:
                del _state["results_status"]["Test loss"]

            if "Mean F1-score" in _state["results_status"]:
                del _state["results_status"]["Mean F1-score"]

            if check_state("classifier_buffer"):
                del _state["classifier_buffer"]

    # ================ MODEL TRAINING AND VALIDATION ================

    if check_state(
        "X_train",
        "y_train",
        "X_val",
        "y_val",
        "X_test",
        "y_test",
        "edges_train",
        "edges_val",
        "edges_test",
        check_all=True,
    ):

        with col2_left:
            with st.spinner(f"Training {classifier_options} model..") as training:
                if classifier_options == "TensorFlow":
                    from models.tesnorfloow import TensorFlowModel as TF

                    tf_constants_dict = TF.to_constants(_state)

                    classifier, history, acc, loss = TF.train(
                        tf_constants_dict["X_train"],
                        tf_constants_dict["y_train"],
                        tf_constants_dict["X_test"],
                        tf_constants_dict["y_test"],
                        parms,
                    )

                    persist("classifier", classifier)
                    persist("history", history)
                    
                    UI.results_status(
                        f"Test accuracy",
                        f"{acc:.4f}",
                    )
                    
                    UI.results_status(
                        "Test loss",
                        f"{loss * 2:.4f}",
                    )

                    test_results, val_results = TF.valid(_state)

                else:

                    from models.sklearn import SkLearn as SK

                    _classifier = SK.classifier(_state)
                    classifier, cv_scores = SK.train(_classifier, _state)

                    persist("classifier", classifier)
                    UI.results_status(
                        "Mean F1-score",
                        f"{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f}",
                    )

                    test_results, val_results = SK.valid(_state)
                
                UI.task_status("Training Model", "✅")
                
                UI.flash_message(
                    "classifier",
                    message=f"{_state.classifier_options} training completed!",
                    col=col1,
                )
                
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
                
    else:
        col2.error(
            "No data available for training. Please ensure you have prepared your dataset on the previous page"
        )


# Display model's download button
with col2_left:
    if check_state("classifier") and _state.previous_classifier == classifier_options:
        download_clasifer(_state["classifier"])
        download_classifier_button()


# Only show page's related results and task status
UI.show_task_status(col3, expand=False)

# Embeddings and data processing results
UI.show_results_status(
    col3,
    stages=["Embeddings length", "Train size", "Test size", "Val size"],
    header="Data Processing Results",
    expand=False,
)

# Model training results
UI.show_results_status(col3, stages=["Test accuracy", "Test loss", "Mean F1-score"], header="Model Training Results",)
# UI.show_results_status(col3, stages=["Selected Model", "(Test Set)", "(Validation Set)"], header="Model Training Results2", )
UI.show_model_results(col3)

# DELETE IN PRODUCTION
manage_state(col3)
