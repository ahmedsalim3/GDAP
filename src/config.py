# ---------------------
# CONFIGURATION
# ---------------------
# Importing necessary libraries
import os

# Set disease ID for cardiovascular disease (EFO-ID)
disease_id = "EFO_0000319"
params = {"disease_id": disease_id}

# Path to BigQuery credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = (
    f"src/open_targets/stemaway-d5a18133ff83.json"
)

# Maximum number of protein-protein interactions (PPI)
max_ppi_interactions = 5000000

# Set ratio of negative to positive samples for classification
negative_to_positive_ratio = 10

# Define data source (can be direct, indirect, or global scores)
data_source = "BigQuery_direct_scores"  # Options: "BigQuery_indirect_scores", "GraphQL_global_scores", "BigQuery_direct_scores"

# Specify train/test split ratio for model validation
test_size = 0.2

# Specify whether to split edges for later prediction
split_edges = True

# Define embedding method (options include various node embedding algorithms)
embedding_mode = "simple_node_embedding"  # Options: 'Node2Vec', 'ProNE', 'GGVec', 'simple_node_embedding'

# Set model selection for classification
model_name = "Gradient_Boosting"  # Options: 'Random_Forest', 'Gradient_Boosting', 'SVM', 'Logistic_Regression'

# Define output directory for models, edges, and embeddings
output_dir = "results/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
