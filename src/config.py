from pathlib import Path

from gdap.datasets.open_targets import INDIRECT_SCORES

PROJ_ROOT = Path(__file__).resolve().parents[1]


# ==========================
#       CONFIGURATION
# ==========================
class Config:
    DISEASE_ID = "EFO_0000319"  # Set disease ID for cardiovascular disease
    DATA_SOURCE = "GraphQLClient"  # Options: "BigQueryClient" | "GraphQLClient"
    QUERY = INDIRECT_SCORES  # SQL query needed for BigQueryClient, Options: DIRECT_SCORES | INDIRECT_SCORES
    PPI_INTERACTIONS = 5000000  # Max number of protein-protein interactions
    NEGATIVE_TO_POSITIVE_RATIO = 10  # Ratio for negative to positive samples
    EMBEDDING_MODE = "GGVec"  # Options: 'Node2Vec', 'ProNE', 'GGVec', 'degree_avg'
    MODEL_NAME = "Logistic_Regression"  # Options: 'Random_Forest', 'Gradient_Boosting', 'SVM', 'Logistic_Regression'
    TEST_SIZE = 0.2  # Train/Test split ratio
    OUTPUT_DIR = PROJ_ROOT / "results/"  # Output folder path to save embeddings, models, predictions, etc.
    CREDENTIALS = (
        PROJ_ROOT / "configs/stemaway-d5a18133ff83.json"
    )  # Credentials path if using BigQueryClient as a data source
