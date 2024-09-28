import streamlit as st
from app.functions import (
    fetch_target_data,
    fetch_ppi_data,
    create_graph,
    generate_embeddings,
    split_data,
)
from app.utils import initialize_session_state, download_graph_files
from app.visualizations import visualize_graph
from app.ui import Home_intro, display_fetched_data, display_status


# Page title
st.set_page_config(
    page_title="Gene-Disease Association Prediction",
    layout="wide",
)

# Initialize session state keys
initialize_session_state()

# --------------------------#
# DATA FETCHING SECTION
# --------------------------#

# Sidebar for data configuration
data_cont = st.sidebar.expander("Configure Target and STRING database", True)
with data_cont:
    col1, col2 = st.columns(2, vertical_alignment="top")
    with col1:
        st.header("Disease Data")
        fetch_target_data()
    with col2:
        st.header("String Database")
        fetch_ppi_data()

# --------------------------#
# GRAPH CONSTRUCTION SECTION
# --------------------------#

# Sidebar for graph configuration
visualize_button, vis_graph_error_flag = False, False
graph_cont = st.sidebar.expander("Configure Graph", True)
with graph_cont:
    st.header("Graph Parameters")
    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        negative_to_positive_ratio = st.number_input(
            "Negative to Positive Ratio:",
            min_value=1,
            max_value=10,
            value=10,
            help="Ratio of negative to positive edges for the graph.",
        )
    with col2:
        # NOTE: This option isn't recommended because it reduces the number of positive edges
        # The positive edges are the minority compared to the overwhelming negative edges from the PPI data
        # This option is added purely for experimental purposes, particularly if the disease genes are abundant
        include_top_positive_checkbox = st.checkbox("Select Top Positives Only?")
        if include_top_positive_checkbox:
            top_positive_percentage = st.slider(
                "Top Positive Percentage:",
                min_value=0,
                max_value=100,
                value=100,
                help="Limit the graph to a percentage of the highest-scoring disease-gene associations.",
            )
        else:
            top_positive_percentage = None

        visualize_checkbox = st.checkbox("Visualize Graph")
        if visualize_checkbox:
            graph_exists = st.session_state.get("graph") is not None
            if not graph_exists:
                vis_graph_error_flag = True
            else:
                sample_size = st.slider(
                    f"Number of nodes associated to {st.session_state['disease_name']}",
                    min_value=300,
                    max_value=1000,
                    value=100,
                    help="Visualize a graph where communities are identified by distinct colors, showing groups of nodes that are more connected to each other.",
                )
                visualize_button = st.button(
                    "Visualize Graph", disabled=not graph_exists
                )

    graph_error_flag = False
    create_graph_button = st.button("Create Graph")
    if create_graph_button:
        graph_error_flag = create_graph(
            top_positive_percentage, negative_to_positive_ratio
        )

# --------------------------#
# EMBEDDINGS SECTION
# --------------------------#

# Sidebar for embeddings
embed_cont = st.sidebar.expander("Feature Engineering", True)
with embed_cont:
    st.header("Generate Edge Features and Node Embeddings")
    embeddings_mode = st.radio(
        "Select Embedding Mode:",
        ["Node2Vec Model", "ProNE Model", "GGVec Model", "Simple Node Embedding"],
        help="While Simple Node Embedding is the fastest option, it may not "
        "provide the best results. We recommend trying either ProNE or "
        "GGVec for improved performance.",
    )
    # embeddings_mode = st.selectbox(
    #     "Select Embedding Mode:",
    #     options=["Node2Vec Model", "ProNE Model", "GGVec Model", "Simple Node Embedding"]
    # )
    embedding_button = st.button("Generate Embedding")

    X, y, embed_error_flag = None, None, False
    if embedding_button:
        X, y, edges, embed_error_flag, embeddings_mode = generate_embeddings(
            embeddings_mode
        )


# -----------#
# DATA SPLIT
# -----------#

# Sidebar for data split
split_ratio_flag = False
model_cont = st.sidebar.expander("Data Spliting", True)
with model_cont:
    split_ratio_flag = split_data(X, y)

if (
    st.session_state.get("X_train") is not None
    and st.session_state.get("y_train") is not None
):
    if st.sidebar.button("Go to Model Training"):
        st.switch_page("pages/1_Model_Training.py")

# -----------------------------------#
# DATA DISPLAY & DOWNLOAD SECTION
# -----------------------------------#

# Introduction
Home_intro()

col1, col2 = st.columns([10, 2])


with col1:
    # Call the display data function to show the data
    display_fetched_data()

    display_status(
        embeddings_mode,
        include_top_positive_checkbox,
        graph_error_flag,
        embed_error_flag,
        split_ratio_flag,
        vis_graph_error_flag,
    )


with col2:
    if st.session_state.get("graph_created", False):
        with st.expander("Download Graph Data", expanded=False):
            download_graph_files()

    if "process_tracker" in st.session_state:
        for stage, status in st.session_state["process_tracker"].items():
            st.write(f"{stage}: {status}")
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

st.markdown(
    f"""
    -----
    """
)

if not graph_error_flag:
    if visualize_button:
        visualize_graph(sample_size=sample_size)
