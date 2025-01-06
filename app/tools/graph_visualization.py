###################################################################################################################################
#  Functions for visualizing a sampled disease-related network using NetworkX (static) and PyVis (interactive) in Streamlit page
###################################################################################################################################

import streamlit as st
from streamlit import session_state as _state
import networkx as nx
from pyvis.network import Network
from networkx.algorithms import community
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import random
import tempfile



def _sample_graph(num_nodes):
    """
    Create a subgraph from the original graph G in session state
    with the specified number of nodes.

    Parameters
    ----------
    num_nodes: The number of nodes to sample and create a subgraph from
    """
    
    G = _state["G"]
    if num_nodes > len(G.nodes):
        raise ValueError("Number of nodes to sample exceeds the total number of nodes in the graph.")

    sampled_nodes = random.sample(list(G.nodes), num_nodes)
    sampled_graph = G.subgraph(sampled_nodes)

    return sampled_graph


def sample_graph(num_nodes):
    """
    Create a subgraph from the original gragh G in session state
    with the specified number of nodes, including the disease node

    Parameters
    ----------
    num_nodes: The number of nodes to sample and create a subgraph from

    """
    G = _state["G"]
    disease_name = _state["disease_name"].lower()

    disease_node = next(
        (n for n in G.nodes if isinstance(n, str) and disease_name in n.lower()), None
    )

    if disease_node:
        remaining_nodes_sample = random.sample(
            [n for n in G.nodes if n != disease_node], min(num_nodes - 1, len(G) - 1)
        )
        sampled_nodes = [disease_node] + remaining_nodes_sample
    else:
        sampled_nodes = random.sample(list(G.nodes), min(num_nodes, len(G)))

    sampled_graph = G.subgraph(sampled_nodes)

    return sampled_graph


def networkx_visualization(num_nodes):
    """
    Visualizes a static networkx sampled graph in Streamlit application using
    NetworkX with community detection (modularity) and edge weights.

    Parameters
    ----------
    num_nodes: The number of nodes to sample and create a subgraph from
    """

    G = sample_graph(num_nodes)
    disease_name = _state["disease_name"].lower()

    communities = community.greedy_modularity_communities(G)
    colors = [0] * G.number_of_nodes()
    for i, comm in enumerate(communities):
        for node in comm:
            colors[list(G.nodes()).index(node)] = i

    plt.figure(figsize=(20, 10))
    pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)

    nx.draw_networkx(
        G,
        pos,
        with_labels=True,
        node_color=colors,
        cmap=plt.cm.jet,
        edge_color="gray",
        node_size=2000,
        arrows=True,
        font_size=10,
        font_weight="bold",
    )

    edge_labels = nx.get_edge_attributes(G, "weight")
    edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
    title = f"Sample graph with {num_nodes} nodes for {disease_name} disease\nOriginal graph has {len(_state["G"])} nodes"
    plt.title(title, fontsize=12, fontweight="bold")
    st.pyplot(plt)


def weight_to_color(weight, min_weight, max_weight):
    """
    Converts a weight value to a color gradient based on its normalized value.

    Returns a rgb color from red (low) to blue (high).
    """
    weight_norm = (weight - min_weight) / (max_weight - min_weight)
    red = int(255 * weight_norm)
    blue = int(255 * (1 - weight_norm))
    green = 255
    return f"rgb({red}, {green}, {blue})"


def pyvis_visualization(num_nodes):
    """
    Visualizes an interactive sample graph in Streamlit application
    using PyVis, with nodes colored based on edge weights to the disease.
    """

    G = sample_graph(num_nodes)
    disease_name = _state["disease_name"].lower()

    # ================ OPTIONAL: use communities ================

    # communities = community.greedy_modularity_communities(G)
    # node_colors = {}
    # for i, comm in enumerate(communities):
    #     for node in comm:
    #         node_colors[node] = i

    disease_node = next(
        (n for n in G.nodes if isinstance(n, str) and disease_name in n.lower()), None
    )

    # weights of the edges connected to the disease node
    disease_weights = {}
    if disease_node:
        for neighbor in G.neighbors(disease_node):
            weight = G[disease_node][neighbor].get("weight")
            disease_weights[neighbor] = weight

    min_weight = min(disease_weights.values())
    max_weight = max(disease_weights.values())

    # ================ Initialize PyVis Network ================
    disease_net = Network(
        height="900px", width="100%", bgcolor="#222222", font_color="white"
    )
    
    # NOTE: you can convert it into PyVis (will not show 'weights')
    # disease_net.from_nx(G)

    # add nodes with color
    for node in G.nodes:
        if not isinstance(node, (str, int)):
            node = str(node)
 
        if node == disease_name:
            color = "#EDF2E9"
            size = 40
        else:
            # ================ Optional: add community-based colorings ================
            # color = f"rgba({(node_colors.get(node, 0) * 50) % 255}, {(node_colors.get(node, 0) * 100) % 255}, 255, 0.6)"
            size = 20

            if node in disease_weights:
                weight = disease_weights[node]
                color = weight_to_color(weight, min_weight, max_weight)
            else:
                color = "#A9A9A9"

        disease_net.add_node(node, color=color, title=node, size=size)

    # add edges with weights
    for u, v, data in G.edges(data=True):
        weight = data.get("weight")
        edge_type = data.get("type")
        title = f"Weight: {weight:.2f}\nType: {edge_type}"
        disease_net.add_edge(u, v, width=weight, title=title)

    # ================ Physics Settings ================

    # Network layout
    disease_net.repulsion(
        node_distance=420,
        central_gravity=0.33,
        spring_length=110,
        spring_strength=0.10,
        damping=0.95,
    )

    # ================ Optional: custom network visualization options ================

    # options = """
    # {
    #     "physics": {
    #         "enabled": true,
    #         "repulsion": {
    #             "nodeDistance": 200
    #         },
    #         "solver": "barnesHut",
    #         "barnesHut": {
    #             "gravitationalConstant": -3000,
    #             "centralGravity": 0.3,
    #             "springLength": 100,
    #             "springConstant": 0.05
    #         }
    #     },
    #     "interaction": {
    #         "zoomView": true,
    #         "navigationButtons": true
    #     }
    # }
    # """
    #
    # disease_net.set_options(options)

    # save network to html, and render in Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file_path = tmp_file.name
        disease_net.save_graph(tmp_file_path)

    with open(tmp_file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    components.html(html_content, height=800)

    return html_content
