from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import pandas as pd
import os


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
