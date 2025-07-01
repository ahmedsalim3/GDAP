###################################################################################################################################
# A set of functions to handle different graph embedding models (Node2Vec, ProNE, GGVec) using Streamlit caching for optimization
###################################################################################################################################

import streamlit as st


@st.cache_resource
def _Node2Vec():
    from gdap.embeddings import Node2Vec

    return Node2Vec(n_components=32, walklen=10, verbose=True)


@st.cache_resource
def _ProNE():
    from gdap.embeddings import ProNE

    return ProNE(n_components=64, step=5, mu=0.2, theta=0.5, exponent=0.75, verbose=True)


@st.cache_resource
def _GGVec():
    from gdap.embeddings import GGVec

    return GGVec(n_components=64, order=3, verbose=True)


def embedding_generator(G, embeddings_mode):
    if embeddings_mode == "Node2Vec Model":
        model = _Node2Vec()
        embeddings = model.fit_transform(G)
        return embeddings

    elif embeddings_mode == "ProNE Model":
        model = _ProNE()
        embeddings = model.fit_transform(G)
        return embeddings

    elif embeddings_mode == "GGVec Model":
        model = _GGVec()
        embeddings = model.fit_transform(G)
        return embeddings

    elif embeddings_mode == "Simple Node Embedding":
        from gdap.embeddings import EmbeddingGenerator

        embeddings = EmbeddingGenerator.simple_node_embedding(G, dim=64)
        return embeddings
