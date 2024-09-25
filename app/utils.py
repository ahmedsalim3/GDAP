from src.open_targets.bigquery_fetcher import BigQueryClient, direct_scores, indirect_scores
from src.open_targets.graphql_fetcher import GraphQLClient
from src.ppi_data import PPIData
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import io
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt

@st.cache_data
def fetch_graphql_data(disease_id):
    graphql_client = GraphQLClient()
    ot_df =  graphql_client.fetch_full_data(disease_id)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name

@st.cache_data
def fetch_bq_direct_scores(params):
    bq_client = BigQueryClient()
    ot_df =  bq_client.execute_query(direct_scores, params)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name

@st.cache_data
def fetch_bq_indirect_scores(params):
    bq_client = BigQueryClient()
    ot_df = bq_client.execute_query(indirect_scores, params)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name

@st.cache_data
def fetch_ppi_db(max_ppi_interactions):
    ppi_data = PPIData(max_ppi_interactions=max_ppi_interactions)
    ppi_df = ppi_data.process_ppi_data()
    return ppi_df

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False).encode('utf-8')

def init_values(key, value=None):
    """
    set a key in st.session_state to a given value if it does not already exist
    session_states: https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state
    """
    if key not in st.session_state:
        st.session_state[key] = value
      
def initialize_session_state():
    # datasets
    init_values('previous_disease_id', None)
    init_values('disease_name', None)
    init_values('ot_df', None)
    init_values('ppi_df', None)
    init_values('ot_csv', None)
    init_values('ppi_csv', None)
    # graphs
    init_values('graph', None)
    init_values("graph_created", False)
    init_values('positive_edges', None)
    init_values('negative_edges', None)
    # embeddings
    init_values('embedding_model', None)
    init_values('embedding', None)
    init_values('X', None)
    init_values('y', None)
    # dat splitting
    init_values("X_train", None)
    init_values("y_train", None)
    init_values("X_val", None)
    init_values("y_val", None)
    init_values("X_test", None)
    init_values("y_test", None)
    # classifier
    init_values("classifier", None)

def prepare_graph_files_in_memory(G, positive_edges, negative_edges):
    # Prepare the files in memory
    st.session_state.edgelist_buffer = io.BytesIO()
    nx.write_edgelist(G, st.session_state.edgelist_buffer, data=True)  # graph.edgelist
    st.session_state.edgelist_buffer.seek(0)
    
    st.session_state.graphml_buffer = io.BytesIO()
    nx.write_graphml(G, st.session_state.graphml_buffer)  # graph.graphml
    st.session_state.graphml_buffer.seek(0)
    
    edges = [(u, v) for u, v in G.edges()]
    edges_df = pd.DataFrame(edges, columns=["source", "destination"])
    st.session_state.edges_csv_buffer = io.BytesIO()
    edges_df.to_csv(st.session_state.edges_csv_buffer, index=False)  # edges.csv
    st.session_state.edges_csv_buffer.seek(0)
    
    st.session_state.positive_edges_buffer = io.BytesIO()
    np.save(st.session_state.positive_edges_buffer, np.array(positive_edges)) # positive_edges.npy
    st.session_state.positive_edges_buffer.seek(0)
    
    st.session_state.negative_edges_buffer = io.BytesIO()
    np.save(st.session_state.negative_edges_buffer, np.array(negative_edges)) # negative_edges.npy
    st.session_state.negative_edges_buffer.seek(0)  
        
        
def download_graph_files():
    st.download_button(
        label="Download Edge List (edgelist.txt)",
        data=st.session_state.edgelist_buffer,
        file_name="graph.edgelist",
        mime="text/plain"
    )
    st.download_button(
        label="Download GraphML (graph.graphml)",
        data=st.session_state.graphml_buffer,
        file_name="graph.graphml",
        mime="application/xml"
    )
    st.download_button(
        label="Download Edges CSV (edges.csv)",
        data=st.session_state.edges_csv_buffer,
        file_name="edges.csv",
        mime="text/csv"
    )
    st.download_button(
        label="Download Positive Edges",
        data=st.session_state.positive_edges_buffer,
        file_name="positive_edges.npy",
        mime="application/octet-stream"
    )
    st.download_button(
        label="Download Negative Edges",
        data=st.session_state.negative_edges_buffer,
        file_name="negative_edges.npy",
        mime="application/octet-stream"
    )


def update_process_tracker(stage, status):
    if 'process_tracker' not in st.session_state:
        st.session_state['process_tracker'] = {}
    st.session_state['process_tracker'][stage] = status

def visualize_graph(sample_size=300):
    G = st.session_state['graph']
    disease_name = st.session_state['disease_name'].lower()
    disease_node = next((n for n in G.nodes if isinstance(n, str) and disease_name in n.lower()), None)
    if disease_node:
        remaining_nodes_sample = random.sample([n for n in G.nodes if n != disease_node], min(sample_size - 1, len(G) - 1))
        sampled_nodes = [disease_node] + remaining_nodes_sample
    else:
        sampled_nodes = random.sample(list(G.nodes), min(sample_size, len(G)))
    sampled_graph = G.subgraph(sampled_nodes)
    communities = community.greedy_modularity_communities(sampled_graph)
    colors = [0] * sampled_graph.number_of_nodes()
    for i, comm in enumerate(communities):
        for node in comm:
            colors[list(sampled_graph.nodes()).index(node)] = i
    
    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(sampled_graph, seed=42, k=0.7, iterations=100)
    nx.draw_networkx(sampled_graph, 
                     pos, with_labels=True, 
                     node_color=colors, 
                     cmap=plt.cm.jet, edge_color="gray", node_size=2000, arrows=True, font_size=10, font_weight="bold")
    
    edge_labels = nx.get_edge_attributes(sampled_graph, 'weight')
    edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()} 
    nx.draw_networkx_edge_labels(sampled_graph, pos, edge_labels=edge_labels, font_color='red')
    title = f"Sample graph with {sample_size} nodes for {disease_name} disease\nOriginal graph has {len(G)} nodes"
    plt.title(title, fontsize=12, fontweight='bold')
    st.pyplot(plt)