# Import configuration
from src.config import *

from src import (
    PPIData,
    BigQueryClient,
    GraphQLClient,
    BiGraph,
    map_nodes,
    features_labels,
    features_labels_edges,
    features_labels_edges_idx,
    features_labels_idx,
    split_data,
    split_edge_data,
    Node2Vec,
    ProNE,
    GGVec,
    EmbeddingGenerator,
    sklearn_models,
    train_model,
    validate_model,
    ModelEvaluation,
    predict,
    prediction_results,
)
from src import DIRECT_SCORES, INDIRECT_SCORES
import joblib
import os

# ==========================
# DATA PROCESSING
# ==========================

# 1. Load and process PPI data
ppi_data = PPIData(max_ppi_interactions=MAX_PPI_INTERACTIONS)
ppi_df = ppi_data.process_ppi_data()
ppi_df.head()

# 2. Fetch data based on the selected data source
if DATA_SOURCE == "GraphQL_global_scores":
    # Fetch global scores using GraphQL - NOTE: This score is approximately equal to the indirect scores from BigQuery
    graphql_client = GraphQLClient()
    ot_df = graphql_client.fetch_full_data(DISEASE_ID)
    print(f"Final disease data shape: {ot_df.shape}")

elif DATA_SOURCE == "BigQuery_direct_scores":
    # Fetch direct scores from BigQuery
    bq_client = BigQueryClient()
    ot_df = bq_client.execute_query(DIRECT_SCORES, PARAMS)
    print(f"Final disease data shape: {ot_df.shape}")

elif DATA_SOURCE == "BigQuery_indirect_scores":
    # Fetch indirect scores from BigQuery
    bq_client = BigQueryClient()
    ot_df = bq_client.execute_query(INDIRECT_SCORES, PARAMS)
    print(f"Final disease data shape: {ot_df.shape}")

ot_df.head()

# ==========================
# GRAPH CREATION
# ==========================

# Create a bipartite graph using the fetched disease data and PPI
G, positive_edges, negative_edges = BiGraph.create_graph(
    ot_df,
    ppi_df,
    negative_to_positive_ratio=NEGATIVE_TO_POSITIVE_RATIO,
    output_dir=OUTPUT_DIR,
)

# Visualize a sample of the created graph
BiGraph.visualize_sample_graph(G, ot_df, node_size=300, output_dir=OUTPUT_DIR)

# ==========================
# EMBEDDINGS
# ==========================

if EMBEDDING_MODE == "simple_node_embedding":
    # Generate simple node embeddings
    embeddings = EmbeddingGenerator.simple_node_embedding(G, dim=64)
    disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
    print(f"\nEmbeddings for node '{disease_name}':\n{embeddings[str(disease_name)]}\n")

elif EMBEDDING_MODE == "Node2Vec":
    # Train Node2Vec model on graph
    n2v_model = Node2Vec(n_components=32, walklen=10)
    print("Training Node2Vec model...")
    embeddings = n2v_model.fit_transform(G)
    node_to_index = map_nodes(G)

    disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
    index = node_to_index[disease_name]
    print(f"\nEmbeddings for node '{disease_name}':\n{embeddings[index]}")

    # Save Node2Vec model and vectors
    save_path = os.path.join(OUTPUT_DIR, disease_name, "embedding_wheel/")
    os.makedirs(save_path, exist_ok=True)
    n2v_model.save(os.path.join(save_path, "n2v_model"))
    n2v_model.save_vectors(os.path.join(save_path, "n2v_wheel_model.bin"))

elif EMBEDDING_MODE == "ProNE":
    # Train ProNE model on graph
    prone_model = ProNE(
        n_components=32, step=5, mu=0.2, theta=0.5, exponent=0.75, verbose=True
    )
    print("Training ProNE model...")
    embeddings = prone_model.fit_transform(G)
    node_to_index = map_nodes(G)

    disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
    index = node_to_index[disease_name]
    print(f"Embeddings for node '{disease_name}':\n{embeddings[index]}")

    # Save ProNE model and vectors
    save_path = os.path.join(OUTPUT_DIR, disease_name, "embedding_wheel/")
    os.makedirs(save_path, exist_ok=True)
    prone_model.save(os.path.join(save_path, "prone_model"))
    ProNE.save_vectors(prone_model, os.path.join(save_path, "prone_wheel_model.bin"))
    print(f"Vectors/model saved to {save_path}")

elif EMBEDDING_MODE == "GGVec":
    # Train GGVec model on graph
    ggvec_model = GGVec(n_components=64, order=3, verbose=True)
    print("Training GGVec model...")
    embeddings = ggvec_model.fit_transform(G)
    node_to_index = map_nodes(G)

    disease_name = [d.split()[0] for d in ot_df["disease_name"].unique()][0]
    index = node_to_index[disease_name]
    print(f"Embeddings for node '{disease_name}':\n{embeddings[index]}")

    # Save GGVec model and vectors
    save_path = os.path.join(OUTPUT_DIR, disease_name, "embedding_wheel/")
    os.makedirs(save_path, exist_ok=True)
    ggvec_model.save(os.path.join(save_path, "ggvec_model"))
    GGVec.save_vectors(ggvec_model, os.path.join(save_path, "ggvec_wheel_model.bin"))
    print(f"Vectors/model saved to {save_path}")

# ==========================------
# FEATURE EXTRACTION AND LABELING
# ==========================------

# Extract features and labels from edges and embeddings
if EMBEDDING_MODE == "simple_node_embedding" and not SPLIT_EDGES:
    # Option 1: Directly get feature labels from edges and embeddings (using simple node embedding function)
    X, y = features_labels(positive_edges, negative_edges, embeddings)
    print(f"Sample from X: {X[0:1]}")
    print(f"Sample from y: {y[0:1]}")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, test_size=TEST_SIZE
    )

elif EMBEDDING_MODE == "simple_node_embedding" and SPLIT_EDGES:
    # Option 2: Split edges for prediction, and get features/labels from split edges (using simple node embedding function)
    X, y, edges = features_labels_edges(
        positive_edges, negative_edges, embeddings, scale_features=False
    )
    print(f"Sample from X: {X[0:1]}")
    print(f"Sample from y: {y[0:1]}")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        edges_train,
        edges_val,
        edges_test,
    ) = split_edge_data(X, y, edges, test_size=TEST_SIZE)

elif EMBEDDING_MODE != "simple_node_embedding" and not SPLIT_EDGES:
    # Option 1: Get feature labels using node indexes for advanced embeddings algo (Node2Vec, ProNE, etc.)
    X, y = features_labels_idx(
        positive_edges, negative_edges, embeddings, node_to_index
    )
    print(f"Sample from X: {X[0:1]}")
    print(f"Sample from y: {y[0:1]}")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, test_size=TEST_SIZE
    )

elif EMBEDDING_MODE != "simple_node_embedding" and SPLIT_EDGES:
    # Option 2: Get feature labels from split edges using node indexes for advanced embeddings algo (Node2Vec, ProNE, etc.)
    X, y, edges = features_labels_edges_idx(
        positive_edges, negative_edges, embeddings, node_to_index, scale_features=False
    )
    print(f"Sample from X: {X[0:1]}")
    print(f"Sample from y: {y[0:1]}")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        edges_train,
        edges_val,
        edges_test,
    ) = split_edge_data(X, y, edges, test_size=TEST_SIZE)

# ==========================------
# MODEL SELECTION AND PREDICTIONS
# ==========================------

# OPTION 1: Fit a classifier model based on selected MODEL_NAME from config.py file
model, cv_scores = train_model(
    sklearn_models[MODEL_NAME], X_train, y_train, model_name=MODEL_NAME
)
test_results, val_results = validate_model(
    model, X_test, y_test, X_val, y_val, threshold=0.5
)

# Print results
print(f"\n{MODEL_NAME} (Test Set)")
for metric, value in test_results.items():
    print(f"{metric}: {value:.4f}")

print(f"\n{MODEL_NAME} (Validation Set):")
for metric, value in val_results.items():
    print(f"{metric}: {value:.4f}")

# Save Model
models_path = os.path.join(OUTPUT_DIR, disease_name, "classifier_models/")
os.makedirs(models_path, exist_ok=True)
joblib.dump(model, os.path.join(models_path, f"{MODEL_NAME}.pkl"))

# # OPTION 2: Define your own model and train it
# model = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
# model_name = "Gradient Boosting" # Optional name
# model, cv_scores= train_model(model, X_train, y_train, model_name=model_name)
# test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5)

# # OPTION 3: Example of simple dense model using sequential api
# from src import train_tf_model
# model, history, acc, loss = train_tf_model(X_train, y_train, X_test, y_test, X_val, y_val, epochs=70)
# test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5, is_tf_model=True)
# model_name="Sequential Model"

# ==========================
# PLOT VALIDATION RESULTS
# ==========================

Evaluation = ModelEvaluation(
    model, X_val, y_val, threshold=0.5, model_name=MODEL_NAME, figsize=(14, 12)
)
Evaluation.plot_evaluation()

# # NOTE: In case of Tensorflow model, use this:
# Evaluation = ModelEvaluation(model, X_val, y_val, threshold=0.5, model_name="Sequential Model", figsize=(14,12),
#                              is_tf_model=True, history=history, history_figsize=(14, 5))
# Evaluation.plot_history()
# Evaluation.plot_evaluation()


# ==========================
# PREDICTION RESULTS
# ==========================

if SPLIT_EDGES:
    associated_proteins, non_associated_proteins = predict(
        model, X_val, edges_val, threshold=0.5
    )

    # Print results for associated proteins
    print("\nAssociated Proteins:")
    for i, (edge, (pred, confidence)) in enumerate(associated_proteins.items()):
        if i < 5:
            disease, protein = edge
            print(
                f"Protein: {protein} (Disease: {disease}), Prediction: {pred}, Confidence Score: {confidence:.4f}"
            )

    # Print results for non-associated proteins
    print("\nNon-Associated Proteins:")
    for j, (edge, (pred, confidence)) in enumerate(non_associated_proteins.items()):
        if j < 5:
            disease, protein = edge
            print(
                f"Protein: {protein} (Disease: {disease}), Prediction: {pred}, Confidence Score: {confidence:.4f}"
            )

    associated_df, non_associated_df = prediction_results(
        associated_proteins, non_associated_proteins, output_dir=OUTPUT_DIR
    )
    print(
        f"Proteins associated/non-associated to {disease_name} are saved to {OUTPUT_DIR}"
    )
