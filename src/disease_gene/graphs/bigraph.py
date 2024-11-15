import networkx as nx
import pandas as pd
import random
import numpy as np
import os
from tqdm import tqdm
import random
from networkx.algorithms import community
import matplotlib.pyplot as plt


class BiGraph:
    @staticmethod
    def create_graph(
        ot_df,
        ppi_df,
        top_positive_percentage=None,
        negative_to_positive_ratio=10,
        output_dir=None,
        streamlit=False,
    ):
        """
        Creates a bipartite networkx graph based on disease-gene associations (from OpenTargets)
        and Protein-Protein Interactions (PPIs) (from STRING Database)

        Parameters:
        -----------
        ot_df :
            OpenTargets data containing disease-gene associations.
            NOTE: This sohould be fetched from either GraphQLClient or BigQueryClient classes
            Should have columns: ['disease_name', 'symbol'] and ['globalScore'/'direct_score'/'indirect_score']

        ppi_df : pandas.DataFrame
            Protein-Protein Interaction (PPI) data containing gene pairs.
            NOTE: This should be returned from PPIData class
            Should have columns: ['GeneName1', 'GeneName2', 'combined_score'].

        top_positive_percentage : float, optional
            Percentage of top positive disease-gene association edges to retain as positive edges,
            specified as a whole number (e.g., 30 for 30%, 20 for 20%).
            This parameter primarilycontrols which positive edges to consider in the final list.
            NOTE: This option isn't recommended if the gene symbols in the disease data are low,
            as it reduces the number of positive edges and considers only the top given percentage.
            The positive edges are typically a minority compared to the overwhelming negative edges
            from the PPI data.
            If None, all positive edges will be used.

        negative_to_positive_ratio : int, optional
            Ratio of negative to positive edges. For every positive edge, this controls how many negative edges
            are generated, maintaining a typical imbalance of 10:1 by default

        output_dir : str, optional (default=None)
            Path to the directory where the graph and edges (positive/negative lists) will be saved for future usages.

        Returns:
        --------
        G : networkx.Graph
            A networkx graph object containing both disease-gene and PPI edges.

        positive_edges : list
            list of positive edges (disease-gene associations) added to the graph

        negative_edges : list
            A list of negative edges (non-associated genes) generated to balance the positive edges
        """
        if streamlit:
            from stqdm import stqdm
            _PROGRESS_BAR = stqdm
        else:
            _PROGRESS_BAR = tqdm     

        # Prepare OpenTargets data
        if "globalScore" in ot_df.columns:
            ot_df.rename(columns={"globalScore": "score"}, inplace=True)
        elif "direct_score" in ot_df.columns:
            ot_df.rename(columns={"direct_score": "score"}, inplace=True)
        elif "indirect_score" in ot_df.columns:
            ot_df.rename(columns={"indirect_score": "score"}, inplace=True)

        ot_df = ot_df.dropna(subset=["symbol", "disease_name"])
        ot_df = ot_df.drop_duplicates(subset=["disease_name", "symbol"])

        # Initialize an empty graph
        G = nx.Graph()

        # Extract unique diseases and prepare disease-gene edges with positive scores
        diseases = [d.split()[0] for d in ot_df["disease_name"].unique()]
        positive_edges = [
            (disease, row["symbol"], {"weight": row["score"], "type": "disease-gene"})
            for disease in diseases
            for _, row in ot_df.iterrows()
            if row["score"] > 0.1  # Keep edges with score > 0.1
        ]
        G.add_edges_from(positive_edges)

        if top_positive_percentage:
            top_n = int((top_positive_percentage / 100) * len(positive_edges))
            positive_edges = sorted(
                positive_edges, key=lambda x: x[2]["weight"], reverse=True
            )[:top_n]
            print(
                f"Number of top positive edges ({top_positive_percentage}%): {len(positive_edges)}"
            )
        else:
            print(f"Number of positive edges: {len(positive_edges)}")

        # Add PPI edges
        ppi_edges_added = 0
        for _, row in _PROGRESS_BAR(
            ppi_df.iterrows(), total=len(ppi_df), desc="Adding PPI edges"
        ):
            if row["GeneName1"] != row["GeneName2"]:
                G.add_edge(
                    row["GeneName1"],
                    row["GeneName2"],
                    weight=row[
                        "combined_score"
                    ],  # add weight derived from combined_score
                    type="ppi",
                )
                ppi_edges_added += 1
        # print(f"Number of PPI edges added: {ppi_edges_added}")

        # Save the graph and export edges
        if output_dir:
            save_path = os.path.join(output_dir, diseases[0], "network")
            os.makedirs(save_path, exist_ok=True)
            nx.write_edgelist(G, os.path.join(save_path, "graph.edgelist"), data=True)
            nx.write_graphml(G, os.path.join(save_path, "graph.graphml"))

            edges = [(u, v) for u, v in G.edges()]
            edges_df = pd.DataFrame(edges, columns=["source", "destination"])
            edges_df.to_csv(os.path.join(save_path, "edges.csv"), index=False)

        # Generate negative edges
        all_genes = set(ppi_df["GeneName1"]).union(set(ppi_df["GeneName2"]))
        negative_edges = []

        for disease in _PROGRESS_BAR(diseases, desc="Generating Negative Edges", total=len(diseases)):
            associated_genes = set(G.neighbors(disease))
            non_associated_genes = all_genes - associated_genes

            # Sort non-associated genes based on combined scores from PPI data
            # so that the generated negative edges are more likely to represent meaningful connections rather than random selections
            potential_negatives = ppi_df[
                ppi_df["GeneName1"].isin(non_associated_genes)
                | ppi_df["GeneName2"].isin(non_associated_genes)
            ]
            potential_negatives = potential_negatives.sort_values(
                by="combined_score", ascending=False
            )

            for _, row in potential_negatives.iterrows():
                gene1, gene2 = row["GeneName1"], row["GeneName2"]
                if disease != gene1 and disease != gene2:
                    negative_edges.append((disease, gene1))
                    negative_edges.append((disease, gene2))

                    if len(negative_edges) >= negative_to_positive_ratio * len(
                        positive_edges
                    ):
                        break

            # Limit the number of negative edges to maintain negative_to_positive_ratio
            if len(negative_edges) > negative_to_positive_ratio * len(positive_edges):
                negative_edges = random.sample(
                    negative_edges, negative_to_positive_ratio * len(positive_edges)
                )

        print(f"Number of negative edges: {len(negative_edges)}")

        # Save positive and negative edges if output_dir is provided
        if output_dir:
            np.save(
                os.path.join(save_path, "positive_edges.npy"), np.array(positive_edges)
            )
            np.save(
                os.path.join(save_path, "negative_edges.npy"), np.array(negative_edges)
            )

        return G, positive_edges, negative_edges

    @staticmethod
    def visualize_sample_graph(
        G, ot_df, node_size=300, figsize=(20, 10), output_dir=None
    ):
        disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
        disease_node = next(
            (n for n in G.nodes if isinstance(n, str) and disease_name in n.lower()),
            None,
        )
        if disease_node:
            remaining_nodes_sample = random.sample(
                [n for n in G.nodes if n != disease_node],
                min(node_size - 1, len(G) - 1),
            )
            sampled_nodes = [disease_node] + remaining_nodes_sample
        else:
            sampled_nodes = random.sample(list(G.nodes), min(node_size, len(G)))
        sampled_graph = G.subgraph(sampled_nodes)
        communities = community.greedy_modularity_communities(sampled_graph)
        colors = [0] * sampled_graph.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                colors[list(sampled_graph.nodes()).index(node)] = i

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(sampled_graph, seed=42, k=0.7, iterations=100)
        nx.draw_networkx(
            sampled_graph,
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

        edge_labels = nx.get_edge_attributes(sampled_graph, "weight")
        edge_labels = {k: f"{v:.3f}" for k, v in edge_labels.items()}
        nx.draw_networkx_edge_labels(
            sampled_graph, pos, edge_labels=edge_labels, font_color="red"
        )
        title = f"Sample graph with {node_size} nodes for {disease_name} disease\nOriginal graph has {len(G)} nodes"
        plt.title(title, fontsize=12, fontweight="bold")
        if output_dir:
            disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
            save_path = os.path.join(output_dir, disease_name, "network")
            output_path = os.path.join(
                save_path, f"{disease_name}_sample_graph_{node_size}_nodes.png"
            )
            plt.savefig(output_path)
            print(f"Sample graph saved to {output_path}")
        plt.show()
