import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def map_nodes(G):
    """Create a mapping from node to its index in the graph"""
    return {node: idx for idx, node in enumerate(G.nodes())}


def edge_features_idx(edge, embeddings, node_index_map):
    """Compute edge features using node index from node-to-index mapping"""
    node1_idx = node_index_map.get(edge[0])
    node2_idx = node_index_map.get(edge[1])

    if node1_idx is None or node2_idx is None:
        return np.zeros(embeddings.shape[1])  # return zero vector if nodes are not found

    return embeddings[node1_idx] * embeddings[node2_idx]


def edge_features(edge, embeddings):
    """Compute edge features directly from embeddings without indexing"""
    return embeddings[edge[0]] * embeddings[edge[1]]


def features_labels_idx(pos_edges, neg_edges, embeddings, node_index_map):
    """Generate features and labels for edges using node index"""
    X_positive = np.array([edge_features_idx(edge, embeddings, node_index_map) for edge in pos_edges])
    X_negative = np.array([edge_features_idx(edge, embeddings, node_index_map) for edge in neg_edges])

    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges))

    return X, y


def features_labels(pos_edges, neg_edges, embeddings):
    """Generate features and labels for edges without using node index"""
    X_positive = np.array([edge_features(edge, embeddings) for edge in pos_edges])
    X_negative = np.array([edge_features(edge, embeddings) for edge in neg_edges])

    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges))

    return X, y


def split_data(X, y, test_size=0.25):
    """Split the dataset into training, validation, and test sets"""
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


def features_labels_edges_idx(pos_edges, neg_edges, embeddings, node_index_map, scale_features=False):
    """Generate edge features, labels, and edge pairs using node index, with optional scaling"""
    pos_edges_with_features = [
        ((edge[0], edge[1]), edge_features_idx(edge, embeddings, node_index_map)) for edge in pos_edges
    ]
    neg_edges_with_features = [
        ((edge[0], edge[1]), edge_features_idx(edge, embeddings, node_index_map)) for edge in neg_edges
    ]

    X_positive = np.array([features for _, features in pos_edges_with_features])
    X_negative = np.array([features for _, features in neg_edges_with_features])

    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges))

    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    edge_pairs = [edge for edge, _ in pos_edges_with_features] + [edge for edge, _ in neg_edges_with_features]

    return X, y, edge_pairs


def features_labels_edges(pos_edges, neg_edges, embeddings, scale_features=False):
    """Generate edge features, labels, and edge pairs without using node index, with optional scaling"""
    pos_edges_with_features = [((edge[0], edge[1]), edge_features(edge, embeddings)) for edge in pos_edges]
    neg_edges_with_features = [((edge[0], edge[1]), edge_features(edge, embeddings)) for edge in neg_edges]

    X_positive = np.array([features for _, features in pos_edges_with_features])
    X_negative = np.array([features for _, features in neg_edges_with_features])

    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(pos_edges) + [0] * len(neg_edges))

    if scale_features:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    edge_pairs = [edge for edge, _ in pos_edges_with_features] + [edge for edge, _ in neg_edges_with_features]

    return X, y, edge_pairs


def split_edge_data(X, y, edges, test_size=0.25):
    """Split the edge dataset into training, validation, and test sets"""
    X_train, X_temp, y_train, y_temp, edges_train, edges_temp = train_test_split(
        X, y, edges, test_size=test_size, random_state=42
    )
    X_val, X_test, y_val, y_test, edges_val, edges_test = train_test_split(
        X_temp, y_temp, edges_temp, test_size=0.5, random_state=42
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, edges_train, edges_val, edges_test
