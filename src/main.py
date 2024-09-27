# Import configuration

from src.config import *

# Import dataset and client modules
from src.ppi_data import PPIData
from src.open_targets.bigquery_fetcher import BigQueryClient, direct_scores, indirect_scores
from src.open_targets.graphql_fetcher import GraphQLClient

# Import graph creation and visualization
from src.bigraph import BiGraph

# Import edge-related feature engineering functions
from src.edge_utils import *

# Import embedding models
from src.embeddings import Node2Vec, ProNE, GGVec, EmbeddingGenerator

# Import model selection functions
from src.ml_models import *
from src.model_evaluation import ModelEvaluation
import joblib

# Import prediction functions
from src.edge_predictions import *



# ---------------------
# DATA PROCESSING
# ---------------------

# 1. Load and process PPI data
ppi_data = PPIData(max_ppi_interactions=max_ppi_interactions)
ppi_df = ppi_data.process_ppi_data()
ppi_df.head()


# 2. Fetch data based on the selected data source

if data_source == "GraphQL_global_scores":
    # (i) Fetch global scores using GraphQL - NOTE: This score is approximately equal to the indirect scores from BigQuery
    graphql_client = GraphQLClient()
    ot_df = graphql_client.fetch_full_data(disease_id)
    print(f'Final disease data shape: {ot_df.shape}')

elif data_source == "BigQuery_direct_scores":
    # (ii) Fetch direct scores from BigQuery
    bq_client = BigQueryClient()
    ot_df = bq_client.execute_query(direct_scores, params)
    print(f'Final disease data shape: {ot_df.shape}')
    
elif data_source == "BigQuery_direct_scores":
     # (iii) Fetch indirect scores from BigQuery
    bq_client = BigQueryClient()
    ot_df = bq_client.execute_query(indirect_scores, params)
    print(f'Final disease data shape: {ot_df.shape}')

ot_df.head()


# ---------------------
# GRAPH CREATION
# ---------------------

# Create a bipartite graph using the fetched disease data and PPI
G, positive_edges, negative_edges = BiGraph.create_graph(
    ot_df,
    ppi_df,
    negative_to_positive_ratio=negative_to_positive_ratio,
    output_dir=output_dir
)


# Visualize a sample of the created graph
BiGraph.visualize_sample_graph(G, ot_df, node_size=300, output_dir=output_dir)

# ---------------------
# EMBEDDINGS
# ---------------------

if embedding_mode == "simple_node_embedding":
    # Generate simple node embeddings 
    embeddings = EmbeddingGenerator.simple_node_embedding(G, dim=64)
    disease_name = [d.split()[0] for d in ot_df['disease_name'].unique()][0]
    print(f"\nEmbeddings for node '{disease_name}':\n{embeddings[str(disease_name)]}\n") # Show embeddings for disease node

elif embedding_mode == "Node2Vec":
    # Train Node2Vec model on graph
    n2v_model = Node2Vec(n_components=32, walklen=10)
    print("Training Node2Vec model...")
    embeddings = n2v_model.fit_transform(G)
    node_to_index = map_nodes(G)
    
    disease_name = [d.split()[0] for d in ot_df['disease_name'].unique()][0]
    index = node_to_index[disease_name]
    print(f"\nEmbeddings for node '{disease_name}':\n{embeddings[index]}")
    
    # Save Node2Vec model and vectors
    save_path = output_dir +  disease_name + '/embedding_wheel/'
    os.makedirs(save_path, exist_ok=True)
    n2v_model.save(save_path + 'n2v_model')
    n2v_model.save_vectors(save_path + "n2v_wheel_model.bin")
    
elif embedding_mode == "ProNE":
    # Train ProNE model on graph
    prone_model = ProNE(
        n_components=32,
        step=5,
        mu=0.2,
        theta=0.5,
        exponent=0.75,
        verbose=True
    )
    # Fit model to graph
    print("Training ProNE model...")
    embeddings = prone_model.fit_transform(G)
    node_to_index = map_nodes(G)

    disease_name = [d.split()[0] for d in ot_df['disease_name'].unique()][0]
    index = node_to_index[disease_name]
    print(f"Embeddings for node '{disease_name}':\n{embeddings[index]}")
    
    # Save ProNE model and vectors
    save_path = output_dir +  disease_name + '/embedding_wheel/'
    os.makedirs(save_path, exist_ok=True)
    prone_model.save(save_path + 'prone_model')
    ProNE.save_vectors(prone_model, save_path + "prone_wheel_model.bin")
    print(f"Vectors/model saved to {save_path}")

    
elif embedding_mode == "GGVec":
    # Train GGVec model on graph
    ggvec_model = GGVec(
        n_components=64,       
        order=3,     
        verbose=True 
    )
    print("Training ProNE model...")
    embeddings = ggvec_model.fit_transform(G)
    node_to_index = map_nodes(G)
    
    disease_name = [d.split()[0] for d in ot_df['disease_name'].unique()][0]
    index = node_to_index[disease_name]
    print(f"Embeddings for node '{disease_name}':\n{embeddings[index]}")
    
    # Save GGVec model and vectors
    save_path = output_dir +  disease_name + '/embedding_wheel/'
    os.makedirs(save_path, exist_ok=True)
    ggvec_model.save(save_path + 'ggvec_model')
    GGVec.save_vectors(ggvec_model, save_path + "ggvec_wheel_model.bin")
    print(f"Vectors/model saved to {save_path}")


# ------------------------------
# FEATURE EXTRACTION AND LABELING
# ------------------------------

# Extract features and labels from edges and embeddings

if embedding_mode == "simple_node_embedding" and not split_edges:
    # Option 1: Directly get feature labels from edges and embeddings (using simple node embedding function)
    X, y = features_labels(positive_edges, negative_edges, embeddings)
    print(f'Sample from X: {X[0:1]}')
    print(f'Sample from y: {y[0:1]}')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=test_size)

elif embedding_mode == "simple_node_embedding" and split_edges:
    # Option 2: Split edges for prediction, and get features/labels from split edges (using simple node embedding function)
    X, y, edges = features_labels_edges(positive_edges, negative_edges, embeddings, scale_features=False)
    print(f'Sample from X: {X[0:1]}')
    print(f'Sample from y: {y[0:1]}')
    X_train, y_train, X_val, y_val, X_test, y_test, edges_train, edges_val, edges_test = split_edge_data(X, y, edges, test_size=test_size)
    

elif embedding_mode != "simple_node_embedding" and not split_edges:
    # Option 1: Get feature labels using node indexes for advanced embeddings algo (Node2Vec, ProNE, etc.)
    X, y = features_labels_idx(positive_edges, negative_edges, embeddings, node_to_index)
    print(f'Sample from X: {X[0:1]}')
    print(f'Sample from y: {y[0:1]}')
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=test_size)
    
elif embedding_mode != "simple_node_embedding" and split_edges:
    # Option 2: Get feature labels from split edges using node indexes for advanced embeddings algo (Node2Vec, ProNE, etc.)
    X, y, edges = features_labels_edges_idx(positive_edges, negative_edges, embeddings, node_to_index, scale_features=False)
    print(f'Sample from X: {X[0:1]}')
    print(f'Sample from y: {y[0:1]}')
    X_train, y_train, X_val, y_val, X_test, y_test, edges_train, edges_val, edges_test = split_edge_data(X, y, edges, test_size=test_size)
    

# ------------------------------
# MODEL SELECTION AND PREDICTIONS
# ------------------------------

# OPTION 1: Fit a classifier model based on selected model_name from model_selection file
model_name = model_name 
model, cv_scores= train_model(models[model_name], X_train, y_train, model_name=model_name)
test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5)

# # OPTION 2: Define your own model and train it
# model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
# model_name = "Gradient Boosting" # Optional
# model, cv_scores= train_model(model, X_train, y_train, model_name=model_name)
# test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5)

# # OPTION 3: Example of simple dense model using sequential api
# model, history, acc, loss = train_tf_model(X_train, y_train, X_test, y_test, X_val, y_val, epochs=70)
# test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5, is_tf_model=True)
# model_name="Sequential Model"

print(f"\n{model_name} (Test Set)")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")
        
print(f"\n{model_name} (Validation Set):")
for metric, value in val_results.items():
    print(f"{metric}: {value:.4f}")
    
# Save Model
models_path = output_dir + disease_name + '/classifier_models/'
os.makedirs(models_path, exist_ok=True)
joblib.dump(model, models_path + f'{model_name}.pkl') # f'{model_name}.keras'  use this incase of TensorFlow model


# ------------------------
# PLOT VALIDATION RESULTS
# ------------------------

Evaluation = ModelEvaluation(model, X_val, y_val, threshold=0.5, model_name=model_name, figsize=(14,12))
Evaluation.plot_evaluation()

# # NOTE: In case of Tensorflow model, use this:
# Evaluation = ModelEvaluation(model, X_val, y_val, threshold=0.5, model_name="Sequential Model", figsize=(14,12),
#                              is_tf_model=True, history=history, history_figsize=(14, 5))
# Evaluation.plot_history()
# Evaluation.plot_evaluation()

# ---------------------
# PREDICTION RESULTS
# ---------------------

if split_edges:
    associated_proteins, non_associated_proteins = predict(model, X_val, edges_val, threshold=0.5)
    # Print results for associated proteins
    print("\nAssociated Proteins:")
    for i, (edge, (pred, confidence)) in enumerate(associated_proteins.items()):
        if i < 5:
            disease, protein = edge
            print(f"Protein: {protein} (Disease: {disease}), Prediction: {pred}, Confidence Score: {confidence:.4f}")

    # Print results for non-associated proteins
    print("\nNon-Associated Proteins:")
    for j, (edge, (pred, confidence)) in enumerate(non_associated_proteins.items()):
        if j < 5:
            disease, protein = edge
            print(f"Protein: {protein} (Disease: {disease}), Prediction: {pred}, Confidence Score: {confidence:.4f}")
            
    associated_df, non_associated_df = prediction_results(associated_proteins, non_associated_proteins, output_dir=output_dir)
    print(f'Proteins associated/non-associated to {disease_name} are saved to {output_dir}')

