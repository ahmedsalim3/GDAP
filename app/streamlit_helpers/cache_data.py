######################################################################################
# Helper functions for app that fetches, load and cache data to make the app faster.
######################################################################################

import streamlit as st


@st.cache_resource
def _GraphQLClient():
    from gene_disease.datasets.open_targets import GraphQLClient

    return GraphQLClient()


@st.cache_resource
def _BigQueryClient():
    from gene_disease.datasets.open_targets import BigQueryClient

    return BigQueryClient(deploy=True)


@st.cache_resource
def fetch_ppi_db(max_ppi_interactions):
    from gene_disease.datasets.string_database import PPIData

    ppi_data = PPIData(max_ppi_interactions=max_ppi_interactions)

    # Fetch and process the data
    ppi_df = ppi_data.process_ppi_data()

    return ppi_df


@st.cache_resource(show_spinner=True)
def _BiGraph():
    from gene_disease.graphs import BiGraph

    return BiGraph()


@st.cache_data
def fetch_graphql_data(disease_id):
    graphql_client = _GraphQLClient()
    ot_df = graphql_client.fetch_full_data(disease_id)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name


@st.cache_data
def fetch_bq_direct_scores(params):
    from gene_disease.datasets.open_targets import DIRECT_SCORES

    bq_client = _BigQueryClient()
    ot_df = bq_client.execute_query(DIRECT_SCORES, params)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name


@st.cache_data
def fetch_bq_indirect_scores(params):
    from gene_disease.datasets.open_targets import INDIRECT_SCORES

    bq_client = _BigQueryClient()
    ot_df = bq_client.execute_query(INDIRECT_SCORES, params)
    disease_name = ot_df.disease_name.iloc[0].split()[0]
    return ot_df, disease_name


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


# @st.cache_data(allow_output_mutation=True)
def create_graph(ot_df, ppi_df, top_pos, neg_to_pos):

    graph = _BiGraph()
    G, pos_edges, neg_edges = graph.create_graph(
        ot_df=ot_df,
        ppi_df=ppi_df,
        top_positive_percentage=top_pos,
        negative_to_positive_ratio=neg_to_pos,
        streamlit=True,
    )
    return G, pos_edges, neg_edges
