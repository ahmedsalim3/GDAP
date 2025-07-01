###################################################################################################################################
# Second page of the Analysis section:
# - Constructs a bipartite graph from the collected datasets.
# - Provides options to download the graph files or visualize a subgraph based on a specified number of nodes.
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
from streamlit_helpers.cache_data import create_graph
from tools.graph_visualization import networkx_visualization, pyvis_visualization
from utils import ui as UI
from utils.graph_files import create_graph_buffers, download_graph_files
from utils.state import check_state, delete_state, manage_state, persist
from typing import Optional

# from utils.style import page_layout

# ==========================
# PAGE LAYOUT AND INTERFACE
# ==========================

# Set the page layout
# page_layout(max_width='80%', padding_top='2rem', padding_right='0rem', padding_left='0rem', padding_bottom='0rem')

st.markdown(
    "<h2 style='text-align: center;  color: black;'>Graph Construction",
    unsafe_allow_html=True,
)

st.write(
    """
    DESCRIPTIVE TEXTS

    -------
    """
)


# Columns setup
col1, col2 = st.columns([8, 2], gap="small", vertical_alignment="top")
# ~parameters/visualization ~status

# ==========================
# PARAMETERS SECTION
# ==========================

# with col1.expander("Graph Parameters", expanded=True):
with col1.container(border=True):
    # st.markdown('<h3 style="text-align: center;">Parameters</h3>', unsafe_allow_html=True)
    # st.markdown("-----")

    # ================ GRAPH PARAMETERS ================

    st.markdown("##### Graph Parameters")
    c1, c2, c3, c4 = st.columns([2, 1, 2, 2], vertical_alignment="top")

    neg_to_pos = c1.number_input(
        "Negative to Positive Ratio:",
        min_value=1,
        max_value=10,
        value=10,
        help="Ratio of negative to positive edges for the graph.",
    )

    top_pos_checkbox = c1.checkbox("Top Positives Only?")
    top_pos: Optional[int] = None
    if top_pos_checkbox:
        top_pos = c1.slider(
            "Top Positive Percentage:",
            min_value=0,
            max_value=100,
            value=100,
            help="Limit the graph to a percentage of the highest-scoring disease-gene associations.",
        )

    if top_pos_checkbox and not check_state("G"):
        col1.warning(
            "Use this option with caution, as reducing positives may lead to a loss of critical data, "
            "especially if the disease has few associations.",
            icon="⚠️",
        )
        col1.info(
            "Note that the PPI database contains the majority of classes; setting this limit with a given high number of PPI interactions "
            "may result in an imbalanced dataset."
        )

    create_graph_button = c2.button("Create Graph")

    static_visualize_button = c3.button("Static Visualization (NetworkX)")
    spyvis_visualize_button = c3.button("Interactive Visualization (PyVis)")

    num_nodes = c4.slider(
        "Number of nodes to visualize",
        min_value=100,
        max_value=500,
        value=200,
        help="Larger graphs with more nodes can be complex to visualize. Limit to 500 nodes for better performance.",
    )


# ==========================
# CREATING GRAPH
# ==========================

if create_graph_button:
    # If there is a re-creation of the graph, delete the previous states
    if check_state(
        "edgelist_buffer",
        "graphml_buffer",
        "edges_csv_buffer",
        "pos_edges_buffer",
        "neg_edges_buffer",
        "G",
        check_all=True,
    ):
        delete_state(
            "edgelist_buffer", "graphml_buffer", "edges_csv_buffer", "pos_edges_buffer", "neg_edges_buffer", "G"
        )

    with col1.status("Constructing disease-gene PPI network...", expanded=True) as creating:
        # Check if both datasets exist
        if check_state("ot_df", "ppi_df", check_all=True):
            G, pos_edges, neg_edges = create_graph(
                ot_df=_state.ot_df,
                ppi_df=_state.ppi_df,
                top_pos=top_pos,
                neg_to_pos=neg_to_pos,
            )
            creating.update(label=f"{_state.disease_name}-gene PPI network was constructed.")

            persist("G", G)
            persist("pos_edges", pos_edges)
            persist("neg_edges", neg_edges)

            UI.flash_message(
                "G",
                message=f"{_state.disease_name} and PPI Bigraph created with {len(pos_edges)} positive edges and {len(neg_edges)} negative edges!",
                col=col1,
            )
            UI.task_status("Graph Constructing", "✅")
            UI.results_status("Graph nodes", len(G))
            UI.results_status("Positive edges", len(pos_edges))
            UI.results_status("Negative edges", len(neg_edges))

        else:
            creating.update(label="Graph creation failed", state="error")
            col1.error("Please fetch both the open target and string datasets first to create the graph.")
            G, pos_edges, neg_edges = None, None, None


# Display download buttons
if check_state("G", "pos_edges", "neg_edges", check_all=True):
    if not check_state(
        "edgelist_buffer", "graphml_buffer", "edges_csv_buffer", "pos_edges_buffer", "neg_edges_buffer", check_all=True
    ):
        if c2.button("Download Graph"):
            with col1.status("Creating graph buffers...") as buffer_creating:
                create_graph_buffers(_state.G, _state.pos_edges, _state.neg_edges)
                buffer_creating.update(label="Graph buffers were created successfully.")
    if check_state(
        "edgelist_buffer", "graphml_buffer", "edges_csv_buffer", "pos_edges_buffer", "neg_edges_buffer", check_all=True
    ):
        with col1.expander("Download Graph Data", expanded=False):
            download_graph_files()

# ================ SUB-GRAPH VISUALIZATION ================

if static_visualize_button:
    if check_state("G"):
        with col1:
            networkx_visualization(num_nodes=num_nodes)
    else:
        col1.error("No graph available to visualize. Please create a graph first.")

if spyvis_visualize_button:
    if check_state("G"):
        with col1:
            html_content = pyvis_visualization(num_nodes=num_nodes)
            persist("html_content", html_content)
    else:
        col1.error("No graph available to visualize. Please create a graph first.")


# Show task status
UI.show_task_status(col2)

UI.show_results_status(col2, header="Disease-Gene PPI Network")

if check_state("html_content"):
    col2.download_button(
        label="Download Pyvis Network",
        data=_state["html_content"],
        file_name=f"{_state['disease_name']}_pyvis_network-{num_nodes}-nodes.html",
        mime="text/html",
        help="Download the interactive pyvis network graph as an HTML file.",
    )

# DELETE IN PRODUCTION
manage_state(col2)
