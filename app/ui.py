import streamlit as st


def Home_intro():
    """Main Home page intro"""

    st.title("**Welcome to 🧬 Gene-Disease Association Prediction**")
    st.caption(
        """
    This app enables users to build a machine learning model that predicts gene-disease associations 
    using data from OpenTargets and the STRING database.
    """
    )
    with st.expander("How to use the app?"):
        st.info(
            """To interact with the app, navigate to the [Open Target Platform](https://platform.opentargets.org/) and obtain a disease EFOID. 
        Enter the EFOID to fetch its corresponding gene data, specify the number of protein-protein interactions (potential negatives), choose your embedding mode, select your preferred model, and train it to predict associations. 
        Lastly, analyze the predictions and play a bit with the threshold to understand the genes associated with your disease!"""
        )


def Model_Training_intro():
    """Model training page intro"""

    col1, col2 = st.columns([10, 2])
    with col1:
        st.title("Model Training")
        st.markdown(
            """
            ⚙️ **Choose a Model:** Select your preferred machine learning model and adjust its hyperparameters.

            📈 **Train the Model:** Initiate training and monitor performance metrics, including accuracy and loss. 

            🔍 **Visualize Results:** Explore decision boundaries on both training and validation datasets. Note that while the test data was included during training, validation data remains unseen by the model.

            🩺 **Diagnose Overfitting:** Assess model performance to identify potential overfitting and tweak settings accordingly for optimal results.
            
            -----
            """
        )
    with col2:
        if (
            st.session_state.get("X_train") is not None
            and st.session_state.get("y_train") is not None
        ):
            st.info(
                f"Training size: {st.session_state['X_train'].shape}, {st.session_state['y_train'].shape}"
            )
            st.info(
                f"Testing size: {st.session_state['X_test'].shape}, {st.session_state['y_test'].shape}"
            )
            st.info(
                f"Validation size: {st.session_state['X_val'].shape}, {st.session_state['y_val'].shape}"
            )


def Predictions_intro():
    """Model prediction page intro"""

    col1, col2 = st.columns([10, 2])
    with col1:
        st.title("Model Prediction")
        st.markdown(
            """
            Here, you can proceed with your evaluation and make predictions using either validation or testing data. Adjust the threshold for predictions and click the Make Prediction button to see the results.
            
            The predictions will be categorized into associated genes and non-associated genes to the given disease.

            
            """
        )

    with col2:
        if (
            st.session_state.get("X_train") is not None
            and st.session_state.get("y_train") is not None
        ):
            st.info(
                f"Training size: {st.session_state['X_train'].shape}, {st.session_state['y_train'].shape}"
            )
            st.info(
                f"Testing size: {st.session_state['X_test'].shape}, {st.session_state['y_test'].shape}"
            )
            st.info(
                f"Validation size: {st.session_state['X_val'].shape}, {st.session_state['y_val'].shape}"
            )


def display_fetched_data():
    """Display open-targets and string's fetched data, with a buttons to download the fetched data"""

    col1, col2 = st.columns([5, 5])
    with col1:
        if "ot_df" in st.session_state and st.session_state.ot_df is not None:
            expander = st.expander(
                f"Click here to see {st.session_state.disease_name} data", expanded=True
            )
            with expander:
                st.dataframe(st.session_state.ot_df, width=1000, height=None)
                disease_name = st.session_state["disease_name"] or "disease"
                st.download_button(
                    label=f"Download {disease_name} data",
                    data=st.session_state.ot_csv,
                    file_name=f"{disease_name}.csv",
                    mime="text/csv",
                )
    with col2:
        if "ppi_df" in st.session_state and st.session_state.ppi_df is not None:
            expander = st.expander("Click here to see PPI data", expanded=True)
            with expander:
                st.dataframe(st.session_state.ppi_df, width=1000, height=None)
                st.download_button(
                    label=f"Download PPI data.csv",
                    data=st.session_state.ppi_csv,
                    file_name="PPI_data.csv",
                    mime="text/csv",
                )


def display_status(
    embeddings_mode,
    include_top_positive_checkbox,
    graph_error_flag,
    embed_error_flag,
    split_ratio_flag,
    vis_graph_error_flag,
):
    """Display the messages status of the preprocessing done in Home page"""

    if (
        st.session_state.get("ot_df") is not None
        and st.session_state.get("ppi_df") is not None
    ):
        st.success("Both Target and PPI data fetched successfully!")
    if st.session_state.get("graph") is not None:
        st.success(
            f"{st.session_state['disease_name']} and PPI Bigraph created with {len(st.session_state['positive_edges'])} positive edges and {len(st.session_state['negative_edges'])} negative edges!"
        )
    if st.session_state.get("X") is not None and st.session_state.get("y") is not None:
        st.success(
            f"{len(st.session_state.embedding)} embeddings Generated using {embeddings_mode}!"
        )
    if (
        st.session_state.get("X_train") is not None
        and st.session_state.get("y_train") is not None
    ):
        st.success(f"Data split successfully!")

    if include_top_positive_checkbox:
        st.warning(
            "Use this option with caution, as reducing positives may lead to a loss of critical data, "
            "especially if the disease has few associations.",
            icon="⚠️",
        )
        st.info(
            "Note that the PPI database contains the majority of classes; setting this limit with a given high number of PPI interactions "
            "may result in an imbalanced dataset."
        )

    if graph_error_flag:
        st.error(
            "Please fetch both the open target and string datasets first to create the graph."
        )
    if embed_error_flag:
        st.error("Please ensure that the graph is available.")
    if split_ratio_flag:
        st.error(
            "Cannot split data because features are not defined, please generate embeddings first."
        )
    if vis_graph_error_flag:
        st.error("No graph available to visualize. Please create a graph first.")


def display_metric_results():
    """Display training status, and metric results. Available from Model Training page"""

    if "results_tracker" in st.session_state:
        for stage, results in st.session_state["results_tracker"].items():
            if "Cross-Validation" in stage:
                output = "\n".join([f"{stage}\n{results}"])
                st.code(output)
            elif "Training Performance" in stage:
                output2 = "\n".join([f"{stage}:\n{results}"])
                st.code(output2)
            elif "Test Set" in stage or "Validation Set" in stage:
                metrics_output = "\n".join(
                    [f"{metric}: {value:.4f}" for metric, value in results.items()]
                )
                st.code(metrics_output)


def display_training_status(classifier_error_flag):
    if "process_tracker" in st.session_state:
        for stage, status in st.session_state["process_tracker"].items():
            st.write(f"{stage}: {status}")

    display_metric_results()

    if (
        not classifier_error_flag
        and st.session_state.get("classifier_buffer") is not None
    ):
        st.download_button(
            label=f"Download {st.session_state.get('classifier_name', 'model')} model",
            data=st.session_state.classifier_buffer,
            file_name=f"{st.session_state.get('classifier_name', 'model')}.pkl",
            mime="application/octet-stream",
        )
    if st.session_state.get("classifier") is not None:
        st.success(f"Selected Model: {st.session_state["classifier"]}")


def display_predicted_data():
    """
    Display predictions of the associated genes and non-associated to the selected disease and download them
    """

    col1, col2 = st.columns([5, 5])
    with col1:
        if (
            "associated_df" in st.session_state
            and st.session_state.associated_df is not None
        ):
            expander = st.expander(
                f"Click here to see associated genes to {st.session_state.disease_name}",
                expanded=True,
            )
            with expander:
                st.dataframe(st.session_state.associated_df, width=1000, height=None)
                disease_name = st.session_state["disease_name"] or "disease"
                st.download_button(
                    label=f"Download associated predection for {disease_name}",
                    data=st.session_state.associated_csv,
                    file_name=f"{disease_name}_associated_predection.csv",
                    mime="text/csv",
                )
    with col2:
        if (
            "non_associated_df" in st.session_state
            and st.session_state.non_associated_df is not None
        ):
            expander = st.expander(
                "Click here to see non-associated genes to {st.session_state.disease_name}",
                expanded=True,
            )
            with expander:
                st.dataframe(
                    st.session_state.non_associated_df, width=1000, height=None
                )
                st.download_button(
                    label=f"Download non-associated predection for {disease_name}",
                    data=st.session_state.non_associated_csv,
                    file_name=f"{disease_name}_non_associated_predection.csv",
                    mime="text/csv",
                )
