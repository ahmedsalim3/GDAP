###################################################################################################################################
#  Utility functions to save graph data in memory buffers and provide download options for various graph formats.
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
from utils import state
import io
import networkx as nx
import pandas as pd
import numpy as np


def create_graph_buffers(G, pos_edges, neg_edges):

    edgelist_buffer = io.BytesIO()
    nx.write_edgelist(G, edgelist_buffer, data=True)
    edgelist_buffer.seek(0)
    state.persist("edgelist_buffer", edgelist_buffer)

    graphml_buffer = io.BytesIO()
    nx.write_graphml(G, graphml_buffer)
    graphml_buffer.seek(0)
    state.persist("graphml_buffer", graphml_buffer)

    edges = [(u, v) for u, v in G.edges()]
    edges_df = pd.DataFrame(edges, columns=["source", "destination"])
    edges_csv_buffer = io.BytesIO()
    edges_df.to_csv(edges_csv_buffer, index=False)
    edges_csv_buffer.seek(0)
    state.persist("edges_csv_buffer", edges_csv_buffer)

    pos_edges_buffer = io.BytesIO()
    np.save(pos_edges_buffer, np.array(pos_edges))
    pos_edges_buffer.seek(0)
    state.persist("pos_edges_buffer", pos_edges_buffer)

    neg_edges_buffer = io.BytesIO()
    np.save(neg_edges_buffer, np.array(neg_edges))
    neg_edges_buffer.seek(0)
    state.persist("neg_edges_buffer", neg_edges_buffer)


def download_graph_files():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.download_button(
        label="Edge List",
        data=_state.edgelist_buffer,
        file_name="graph.edgelist",
        mime="text/plain",
        help="graph.edgelist",
    )
    c2.download_button(
        label="Edges CSV",
        data=_state.edges_csv_buffer,
        file_name="edges.csv",
        mime="text/csv",
        help="edges.csv",
    )
    c3.download_button(
        label="GraphML Format",
        data=_state.graphml_buffer,
        file_name="graph.graphml",
        mime="application/xml",
        help="graph.graphml",
    )
    c4.download_button(
        label="Positive Edges",
        data=_state.pos_edges_buffer,
        file_name="positive_edges.npy",
        mime="application/octet-stream",
        help="positive_edges.npy",
    )
    c5.download_button(
        label="Negative Edges",
        data=_state.neg_edges_buffer,
        file_name="negative_edges.npy",
        mime="application/octet-stream",
        help="negative_edges.npy",
    )
