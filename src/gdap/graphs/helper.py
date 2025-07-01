import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
from networkx.algorithms import community


def fetch_OT_data_from_api(url, query, variables):
    try:
        response = requests.post(url, json={"query": query, "variables": variables})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def process_OT_data(api_res):
    associated_targets = api_res["data"]["disease"]["associatedTargets"]["rows"]
    df = pd.json_normalize(
        associated_targets,
        record_path="datasourceScores",
        meta=["target", "score"],
        record_prefix="datasourceScores_",
        errors="ignore",
    )
    target_df = pd.json_normalize(df["target"])
    df = df.drop(columns=["target"])
    return pd.concat([df, target_df], axis=1)


def aggregate_OT_data(df):
    agg_df = (
        df.groupby("id")
        .agg(
            {
                "approvedSymbol": "first",
                "score": "mean",
                "datasourceScores_id": lambda x: list(x),
                "datasourceScores_score": lambda x: list(x),
            }
        )
        .reset_index()
    )

    gene_symbols = agg_df["approvedSymbol"].tolist()
    features = []
    for feature_list in agg_df["datasourceScores_id"].tolist():
        for feature in feature_list:
            if feature not in features:
                features.append(feature)
    return agg_df, gene_symbols, features


def fetch_ST_data(genes):
    string_api_url = "https://version-11-5.string-db.org/api"
    output_format = "json"
    method = "network"
    request_url = "/".join([string_api_url, output_format, method])

    params = {
        "identifiers": "%0d".join(genes),
        "species": 9606,  # Human taxonomy, aka Homo sapiens
        "caller_identity": "app.name",
    }
    try:
        response = requests.post(request_url, data=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def process_ST_data(data):
    df = pd.json_normalize(data, errors="ignore")
    # df = data[data['escore'] > 0.4]
    # df = data.sort_values(by='escore', ascending=False)
    return df


def construct_graph(df, source_col, target_col, edge_attr_col, plot=False, save=False, filename=None, figsize=(10, 8)):
    G = nx.from_pandas_edgelist(
        df,
        source=source_col,
        target=target_col,
        edge_attr=edge_attr_col,
        create_using=nx.Graph(),  # undirected Graph
    )

    if plot:
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightgreen",
            edge_color="gray",
            node_size=3000,
            font_size=10,
            font_weight="bold",
            arrows=True,
        )
        edge_labels = nx.get_edge_attributes(G, edge_attr_col)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.title(f"Network for ({G})")
        if save:
            if filename is None:
                filename = f"assets/{G}.png"
            plt.savefig(filename)
        plt.show()

    return G


def plot_community_detection(G, edge_attr_col, plot=True, save=False, filename=None, figsize=(30, 20)):
    if plot or save:
        communities = community.greedy_modularity_communities(G)
        colors = [0] * G.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                colors[list(G.nodes()).index(node)] = i

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)
        nx.draw(
            G, pos, with_labels=True, node_color=colors, cmap=plt.cm.jet, node_size=4000, edge_color="gray", arrows=True
        )
        edge_labels = nx.get_edge_attributes(G, edge_attr_col)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        plt.title(f"Composed Network with Communities\n{G}", fontsize=30, fontweight="bold")
        if save:
            if filename is None:
                filename = "assets/Composed_Network_with_Communities.png"
            plt.savefig(filename)
        plt.show()


def visualize_graphs(G, plot=True, save=False, filename=None, figsize=(30, 20)):
    if plot or save:
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)
        nx.draw_networkx(G, pos, with_labels=True, node_size=4000, font_size=10, font_weight="bold", edge_color="gray")
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="red")
        if filename:
            plt.title(f"{filename} Composed Network\n{G}", fontsize=30, fontweight="bold")
        else:
            plt.title(f"Composed Network\n{G}", fontsize=30, fontweight="bold")
        if save:
            if filename is None:
                filename = "assets/Composed_Network.png"
            plt.savefig(filename)
        plt.show()
