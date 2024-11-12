# ==========================
# CONFIGURATION
# ==========================

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Project root path
PROJ_ROOT = Path(__file__).resolve().parents[1]

# ==========================
#         DATASETS
# ==========================

# 1. Open-targets

# Set disease ID for cardiovascular disease (EFO-ID)
DISEASE_ID = "EFO_0000319"
PARAMS = {"disease_id": DISEASE_ID}

# Define data source (can be direct, indirect, or global scores)
DATA_SOURCE = "BigQuery_direct_scores"  # Options: "BigQuery_indirect_scores", "GraphQL_global_scores", "BigQuery_direct_scores"

# 2. String-database

# Maximum number of protein-protein interactions (PPI)
MAX_PPI_INTERACTIONS = 100

# ==========================
#         GRAPHS
# ==========================

# Set ratio of negative to positive samples for classification
NEGATIVE_TO_POSITIVE_RATIO = 10

# ==========================
#    TRAINING PARAMETERS
# ==========================

# Define embedding method (options include various node embedding algorithms)
EMBEDDING_MODE = "simple_node_embedding"  # Options: 'Node2Vec', 'ProNE', 'GGVec', 'simple_node_embedding'

# Set model selection for classification
MODEL_NAME = "Gradient_Boosting"  # Options: 'Random_Forest', 'Gradient_Boosting', 'SVM', 'Logistic_Regression'

# Specify train/test split ratio for model validation
TEST_SIZE = 0.2

# Specify whether to split edges for later prediction
SPLIT_EDGES = True

# ==========================
#     OUTPUT FOLDER
# ==========================

# Define the output directory to store the dataset, model, graphs, edges, embeddings, and predictions
OUTPUT_DIR = "results/"
if not Path(OUTPUT_DIR).exists():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ==========================
#         SECRETS
# ==========================

# From .env file
CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Or directly using environ
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "configs/stemaway-d5a18133ff83.json"
