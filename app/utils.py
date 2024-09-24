from src.open_targets.bigquery_fetcher import BigQueryClient, direct_scores, indirect_scores
from src.open_targets.graphql_fetcher import GraphQLClient
from src.ppi_data import PPIData
import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import io
     
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
