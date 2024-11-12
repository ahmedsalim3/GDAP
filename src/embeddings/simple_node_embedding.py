import numpy as np


class EmbeddingGenerator:
    @staticmethod
    def simple_node_embedding(G, dim=64):
        print("Generating simple node embeddings...")
        embeddings = {}
        for node in G.nodes():
            # Use node degree as a feature
            degree = G.degree(node)
            # Use average neighbor degree as another feature
            neighbor_degrees = [G.degree(n) for n in G.neighbors(node)]
            avg_neighbor_degree = np.mean(neighbor_degrees) if neighbor_degrees else 0
            # Create a simple embedding vector
            embedding = np.zeros(dim)
            embedding[0] = degree
            embedding[1] = avg_neighbor_degree
            # Fill the rest with random values (you could add more graph-based features here)
            embedding[2:] = np.random.randn(dim - 2)
            embeddings[node] = embedding / np.linalg.norm(embedding)  # Normalize

        return embeddings
