import networkx as nx
import pandas as pd
import random
import numpy as np
import os
from tqdm import tqdm



def create_graph(
    ot_df,
    ppi_df,
    top_positive_percentage=None,
    desired_ratio=10,
    output_dir=None
):
    
    """
    Creates a bipartite networkx graph based on disease-gene associations (from OpenTargets) 
    and Protein-Protein Interactions (PPIs) from PPIData.
    
    Parameters:
    -----------
    ot_df : pandas.DataFrame
        OpenTargets data containing disease-gene associations.
        NOTE: This sohould be fetched from bigquery_fetcher or bigquery_fetcher python files
        Should have columns: ['disease_name', 'symbol', 'globalScore'/'direct_score'/'indirect_score']
        
    ppi_df : pandas.DataFrame
        Protein-Protein Interaction (PPI) data containing gene pairs.
        NOTE: This should be returned from String_DB_PPI class
        Should have columns: ['GeneName1', 'GeneName2', [combined_score].
        
    top_positive_percentage : float, optional
        Percentage of top positive disease-gene association edges to retain, based on their scores.
        If None, all positive edges will be used.
        
    desired_ratio : int, optional
        Ratio of negative to positive edges. For every positive edge, this controls how many negative edges 
        are generated, maintaining a typical imbalance of 10:1 by default
        
    output_dir : str, optional (default=None)
        Path to the directory where the graph and edges (positive/negative) will be saved.
        
    Returns:
    --------
    G : networkx.Graph
        A networkx graph object containing both disease-gene and PPI edges.
        
    positive_edges : list
        list of positive edges (disease-gene associations) added to the graph
        
    negative_edges : list
        A list of negative edges (non-associated genes) generated to balance the positive edges
    """
    
    # Prepare OpenTargets data
    if 'globalScore' in ot_df.columns:
        ot_df.rename(columns={'globalScore': 'score'}, inplace=True)
    elif 'direct_score' in ot_df.columns:
        ot_df.rename(columns={'direct_score': 'score'}, inplace=True)
    elif 'indirect_score' in ot_df.columns:
        ot_df.rename(columns={'indirect_score': 'score'}, inplace=True)
        
    # Initialize an empty graph
    G = nx.Graph()
    
    # Extract unique diseases and prepare disease-gene edges with positive scores
    diseases = [d.split()[0] for d in ot_df['disease_name'].unique()]
    positive_edges = [
        (disease, row['symbol'], {'weight': row['score'], 'type': 'disease-gene'})
        for disease in diseases
        for _, row in ot_df.iterrows() if row['score'] > 0.1  # Keep edges with score > 0.1
    ]
    G.add_edges_from(positive_edges)
    
    # Filter top positive edges if specified
    if top_positive_percentage:
        top_n = int((top_positive_percentage / 100) * len(positive_edges))
        positive_edges = sorted(positive_edges, key=lambda x: x[2]['weight'], reverse=True)[:top_n]
        print(f"Number of top positive edges ({top_positive_percentage}%): {len(positive_edges)}")
    else:
        print(f"Number of positive edges: {len(positive_edges)}")
        
    # Add PPI edges
    ppi_edges_added = 0
    for _, row in tqdm(ppi_df.iterrows(), total=len(ppi_df), desc="Adding PPI edges"):
        if row['GeneName1'] != row['GeneName2']:
            G.add_edge(row['GeneName1'],
                       row['GeneName2'],
                       weight=row['combined_score'],
                       type='ppi')
            ppi_edges_added += 1
    # print(f"Number of PPI edges added: {ppi_edges_added}")
    
    # Save the graph and export edges
    if output_dir:
        save_path = os.path.join(output_dir, diseases[0])
        os.makedirs(save_path, exist_ok=True)
        nx.write_edgelist(G, os.path.join(save_path, "graph.edgelist"), data=True)
        nx.write_graphml(G, os.path.join(save_path, "graph.graphml"))
        
        # Save edges as CSV
        edges = [(u, v) for u, v in G.edges()]
        edges_df = pd.DataFrame(edges, columns=["source", "destination"])
        edges_df.to_csv(os.path.join(save_path, "edges.csv"), index=False)
        
    # Generate negative edges
    all_genes = set(ppi_df['GeneName1']).union(set(ppi_df['GeneName2']))
    negative_edges = []

    for disease in diseases:
        associated_genes = set(G.neighbors(disease))
        non_associated_genes = all_genes - associated_genes
        disease_negative_edges = [(disease, gene) for gene in non_associated_genes]
        negative_edges.extend(disease_negative_edges)
        
    # Sample negative edges based on the desired ratio
    num_negative_samples = min(len(negative_edges), desired_ratio * len(positive_edges))
    negative_edges = random.sample(negative_edges, num_negative_samples)
    print(f"Number of negative edges: {len(negative_edges)}")
    
    # Save positive and negative edges if output_dir is provided
    if output_dir:
        np.save(os.path.join(save_path, "positive_edges.npy"), np.array(positive_edges))
        np.save(os.path.join(save_path, "negative_edges.npy"), np.array(negative_edges))

    return G, positive_edges, negative_edges
