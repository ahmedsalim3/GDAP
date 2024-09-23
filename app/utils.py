from src.open_targets.bigquery_fetcher import BigQueryClient, direct_scores, indirect_scores
from src.open_targets.graphql_fetcher import GraphQLClient
from src.ppi_data import PPIData
import streamlit as st
     
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