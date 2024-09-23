import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential


def get_edge_features_by_index(edge, embeddings, node_to_index):
    """
    Get edge features using node-to-index mapping.

    Returns:
        A numpy array representing the edge features, which is the element-wise product of the node embeddings
    """
    idx1 = node_to_index.get(edge[0])
    idx2 = node_to_index.get(edge[1])
    if idx1 is None or idx2 is None:
        return np.zeros(embeddings.shape[1])  # returns zero vector if nodes not found
    return embeddings[idx1] * embeddings[idx2]


def get_edge_features_direct(edge, embeddings):
    """
    Get edge features directly from embeddings
    
    Returns:
        np.ndarray: Element-wise product of the embeddings for the given edge
    """
    return embeddings[edge[0]] * embeddings[edge[1]]


def split_data(X, y, split_ratio=0.25):
    """Split dataset into training, validation, and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


def split_edge(X, y, edges, split_ratio=0.25):
    """Split edge dataset into training, validation, and test sets"""
    X_train, X_temp, y_train, y_temp, edges_train, edges_temp = train_test_split(
        X, y, edges, test_size=split_ratio, random_state=42
    )    
    X_val, X_test, y_val, y_test, edges_val, edges_test = train_test_split(
        X_temp, y_temp, edges_temp, test_size=0.5, random_state=42
    )
    return X_train, y_train, X_val, y_val, X_test, y_test, edges_train, edges_val, edges_test


def prepare_pred_features(positive_edges, negative_edges, embeddings, node_to_index, scaling=False):
    """
    Prepare edge features and labels for predictions
    """
    positive_edges_with_proteins = [
        ((edge[0], edge[1]), get_edge_features_by_index(edge, embeddings, node_to_index)) for edge in positive_edges
    ]

    negative_edges_with_proteins = [
        ((edge[0], edge[1]), get_edge_features_by_index(edge, embeddings, node_to_index)) for edge in negative_edges
    ]

    X_positive = np.array([feats for _, feats in positive_edges_with_proteins])
    X_negative = np.array([feats for _, feats in negative_edges_with_proteins])
    
    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(positive_edges_with_proteins) + [0] * len(negative_edges_with_proteins))

    if scaling:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    edges = [edge for edge, _ in positive_edges_with_proteins] + [edge for edge, _ in negative_edges_with_proteins]

    return X, y, edges


def generate_edge_features(positive_edges, negative_edges, embeddings, node_to_index):
    """
    Generate features and labels from positive and negative edges.
    """
    X_positive = np.array([get_edge_features_by_index(edge, embeddings, node_to_index) for edge in positive_edges])
    X_negative = np.array([get_edge_features_by_index(edge, embeddings, node_to_index) for edge in negative_edges])
    
    X = np.vstack((X_positive, X_negative))
    y = np.array([1] * len(positive_edges) + [0] * len(negative_edges))

    return X, y



def predict(model, X_val, edges_val, threshold=0.5):
    """
    Predict edge associations using the given model.
    """
    if isinstance(model, Sequential):
        # TensorFlow model predictions
        y_proba = model.predict(X_val) # predicted probs
        confidence_scores = y_proba.flatten() 
        
        # plt.hist(confidence_scores, bins=50)
        # plt.title("Confidencw score")
        # plt.show()
        
        y_pred = (confidence_scores > threshold).astype(int)
    
    else:
        y_proba = model.predict_proba(X_val)[:, 1] # probs for positive class
        y_pred = (y_proba > threshold).astype(int)   # apply the threshold
        confidence_scores = y_proba

    protein_confidence_mapping = {
        edge: (pred.item(), confidence)
        for edge, pred, confidence in zip(edges_val, y_pred, confidence_scores)
    }

    # separate associated and non-associated proteins
    associated_proteins = {edge: (pred, confidence) for edge, (pred, confidence) in protein_confidence_mapping.items() if pred == 1} # asssociated with the disease
    non_associated_proteins = {edge: (pred, confidence) for edge, (pred, confidence) in protein_confidence_mapping.items() if pred == 0} # not associated to the disease
    
    return associated_proteins, non_associated_proteins


def prediction_results(associated_proteins, non_associated_proteins, output_dir=None):
    associated_df = pd.DataFrame(
        [(disease, protein, prediction, confidence) 
         for (disease, protein), (prediction, confidence) in associated_proteins.items()],
        columns=['disease_name', 'symbol', 'prediction', 'confidence']
    )

    non_associated_df = pd.DataFrame(
        [(disease, protein, prediction, confidence) 
         for (disease, protein), (prediction, confidence) in non_associated_proteins.items()],
        columns=['disease_name', 'symbol', 'prediction', 'confidence']
    )
    
    # sort by confidence
    associated_df.sort_values(by='confidence', ascending=False, inplace=True)
    non_associated_df.sort_values(by='confidence', ascending=False, inplace=True)
    
    if output_dir:
        save_path = os.path.join(output_dir, associated_df.disease_name.iloc[0], 'predictions')
        os.makedirs(save_path, exist_ok=True)
        associated_df.to_csv(os.path.join(save_path, 'associated_prediction_results.csv'), index=False)
        non_associated_df.to_csv(os.path.join(save_path, 'non_associated_results.csv'), index=False)

    return associated_df, non_associated_df

