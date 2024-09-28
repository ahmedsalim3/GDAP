import streamlit as st
from src.bigraph import BiGraph
from src.embeddings import *
from src.edge_utils import (
    map_nodes,
    features_labels_edges_idx,
    features_labels_edges,
    split_edge_data,
)
from app.utils import (
    fetch_bq_direct_scores,
    fetch_bq_indirect_scores,
    fetch_graphql_data,
    fetch_ppi_db,
    convert_df,
    update_process_tracker,
    prepare_graph_files_in_memory,
)


def fetch_target_data():
    """Fetch target disease data based on user input."""
    disease_id = st.text_input("Enter Disease EFO-ID:")

    params = {"disease_id": disease_id}
    data_source = st.selectbox(
        "Select Data Source:",
        options=["GraphQL", "Direct Scores (BigQuery)", "Indirect Scores (BigQuery)"],
    )

    fetch_ot = st.button("Fetch Target Data")
    if fetch_ot:
        # Clear previous data if disease EFO-ID changes
        if st.session_state.previous_disease_id != disease_id:
            st.session_state.ot_df = None
            st.session_state.ppi_df = None
            st.session_state.previous_disease_id = disease_id

        if disease_id:
            try:
                # Fetch data based on the selected data source
                if data_source == "GraphQL":
                    st.session_state.ot_df, st.session_state.disease_name = (
                        fetch_graphql_data(disease_id)
                    )
                elif data_source == "Direct Scores (BigQuery)":
                    st.session_state.ot_df, st.session_state.disease_name = (
                        fetch_bq_direct_scores(params)
                    )
                elif data_source == "Indirect Scores (BigQuery)":
                    st.session_state.ot_df, st.session_state.disease_name = (
                        fetch_bq_indirect_scores(params)
                    )

                # Check if data was retrieved successfully
                if (
                    st.session_state.ot_df is not None
                    and not st.session_state.ot_df.empty
                ):
                    st.session_state.ot_csv = convert_df(st.session_state.ot_df)
                    update_process_tracker("Fetching Data", "✔️ Completed")
                else:
                    st.error("No data found for the given EFO-ID.")
                    update_process_tracker("Fetching Data", "❌ Failed")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                update_process_tracker("Fetching Data", "❌ Failed")
        else:
            st.error("Please enter a valid EFO-ID.")


def fetch_ppi_data():
    """Fetch protein-protein interaction data based on user input."""
    max_ppi_interactions = st.slider(
        "Maximum number of PPI interactions",
        min_value=50,
        max_value=5000000,
        step=500,
        value=2500000,
    )
    fetch_ppi = st.button("Fetch PPI Data")
    if fetch_ppi:
        st.session_state.ppi_df = fetch_ppi_db(max_ppi_interactions)
        st.session_state.ppi_csv = convert_df(
            st.session_state.ppi_df
        )  # Save the PPI CSV in session state
        update_process_tracker("Fetching PPI Data", "✔️ Completed")


def create_graph(top_positive_percentage, negative_to_positive_ratio):

    graph_error_flag = False
    if (
        st.session_state.get("ot_df") is not None
        and st.session_state.get("ppi_df") is not None
    ):
        G, positive_edges, negative_edges = BiGraph.create_graph(
            ot_df=st.session_state["ot_df"],
            ppi_df=st.session_state["ppi_df"],
            top_positive_percentage=top_positive_percentage,
            negative_to_positive_ratio=negative_to_positive_ratio,
        )
        st.session_state["graph"] = G
        st.session_state["positive_edges"] = positive_edges
        st.session_state["negative_edges"] = negative_edges
        st.session_state.graph_created = True
        prepare_graph_files_in_memory(G, positive_edges, negative_edges)
        update_process_tracker("Creating Graph", "✔️ Completed")
    else:
        graph_error_flag = True
    return graph_error_flag


def generate_embeddings(embeddings_mode):
    """
    Generate embeddings for the graph based on the selected embedding mode.

    The current parameters for the embedding models are fixed and cannot be
    changed in the Streamlit interface. They are set to approximate minimum
    values for faster embeddings generation.

    Args;
        embeddings_mode (str): The mode of embedding to be used
            - 'Node2Vec Model'
            - 'ProNE Model'
            - 'GGVec Model'
            - 'Simple Node Embedding'
    """
    embed_error_flag = False  # error flag

    if "graph" in st.session_state and "ot_df" in st.session_state:
        G = st.session_state["graph"]
        ot_df = st.session_state["ot_df"]
        positive_edges = st.session_state["positive_edges"]
        negative_edges = st.session_state["negative_edges"]

        if G is None or len(G.nodes()) == 0:
            embed_error_flag = True
            return None, None, None, embed_error_flag, None

        if embeddings_mode == "Node2Vec Model":
            N2V_model = Node2Vec(n_components=32, walklen=10)
            embedding = N2V_model.fit_transform(G)
            st.session_state["embedding_model"] = N2V_model
            st.session_state["embedding"] = embedding

            # Generate edge features
            node_to_index = map_nodes(G)
            st.session_state["node_to_index"] = node_to_index
            X, y, edges = features_labels_edges_idx(
                positive_edges,
                negative_edges,
                embedding,
                node_to_index,
                scale_features=False,
            )
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["edges"] = edges

            # DEBUG
            diseases = [d.split()[0] for d in ot_df["disease_name"].unique()]
            index = node_to_index[diseases[0]]
            print(f"Embeddings for node '{diseases[0]}':\n{embedding[index]}")
            print(f"Sample from X: {X[0:1]}")
            print(f"Sample from y: {y[0:1]}")

            update_process_tracker("Generating Node Embeddings", "✔️ Completed")
            return X, y, edges, embed_error_flag, embeddings_mode

        elif embeddings_mode == "ProNE Model":
            prone_model = ProNE(
                n_components=64, step=5, mu=0.2, theta=0.5, exponent=0.75, verbose=True
            )
            embedding = prone_model.fit_transform(G)
            st.session_state["embedding_model"] = prone_model
            st.session_state["embedding"] = embedding

            # Generate edge features
            node_to_index = map_nodes(G)
            X, y, edges = features_labels_edges_idx(
                positive_edges,
                negative_edges,
                embedding,
                node_to_index,
                scale_features=False,
            )
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["edges"] = edges

            # DEBUG
            diseases = [d.split()[0] for d in ot_df["disease_name"].unique()]
            index = node_to_index[diseases[0]]
            print(f"Embeddings for node '{diseases[0]}':\n{embedding[index]}")
            print(f"Sample from X: {X[0:1]}")
            print(f"Sample from y: {y[0:1]}")

            update_process_tracker("Generating Node Embeddings", "✔️ Completed")
            return X, y, edges, embed_error_flag, embeddings_mode

        elif embeddings_mode == "GGVec Model":
            ggvec_model = GGVec(n_components=64, order=3, verbose=True)
            embedding = ggvec_model.fit_transform(G)
            st.session_state["embedding_model"] = ggvec_model
            st.session_state["embedding"] = embedding

            # Generate edge features
            node_to_index = map_nodes(G)
            X, y, edges = features_labels_edges_idx(
                positive_edges,
                negative_edges,
                embedding,
                node_to_index,
                scale_features=False,
            )
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["edges"] = edges

            # DEBUG
            diseases = [d.split()[0] for d in ot_df["disease_name"].unique()]
            index = node_to_index[diseases[0]]
            print(f"Embeddings for node '{diseases[0]}':\n{embedding[index]}")
            print(f"Sample from X: {X[0:1]}")
            print(f"Sample from y: {y[0:1]}")

            update_process_tracker("Generating Node Embeddings", "✔️ Completed")
            return X, y, edges, embed_error_flag, embeddings_mode

        elif embeddings_mode == "Simple Node Embedding":
            embedding = EmbeddingGenerator.simple_node_embedding(G, dim=64)
            X, y, edges = features_labels_edges(
                positive_edges, negative_edges, embedding, scale_features=False
            )
            st.session_state["X"] = X
            st.session_state["y"] = y
            st.session_state["edges"] = edges
            st.session_state["embedding"] = embedding

            # DEBUG
            print(f"Sample from X: {X[0:1]}")
            print(f"Sample from y: {y[0:1]}")
            st.session_state["X"] = X
            st.session_state["y"] = y
            update_process_tracker("Generating Node Embeddings", "✔️ Completed")
            return X, y, edges, embed_error_flag, embeddings_mode


def split_data(X, y):
    """Split the data into training and test sets."""
    test_size = st.number_input(
        "Test Split Ratio:",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        format="%.2f",  # Format to two decimal places
        help="Split ratio for training, validation, and testing."
        "By default, the test and validation data will be half split.",
    )
    test_size_button = st.button("Test Size")
    if test_size_button:
        X = st.session_state["X"]
        y = st.session_state["y"]
        edges = st.session_state["edges"]
        if X is not None and y is not None and edges is not None:
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
            update_process_tracker("Data splitting", "✔️ Completed")
            st.session_state["X_train"] = X_train
            st.session_state["y_train"] = y_train
            st.session_state["X_val"] = X_val
            st.session_state["y_val"] = y_val
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["edges_train"] = edges_train
            st.session_state["edges_val"] = edges_val
            st.session_state["edges_test"] = edges_test
        else:
            return True
