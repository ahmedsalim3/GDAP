#############
#  DATASET  #
#############
from .datasets.open_targets.bigquery_client import BigQueryClient
from .datasets.open_targets import INDIRECT_SCORES, DIRECT_SCORES
from .datasets.open_targets import GraphQLClient
from .datasets.open_targets import save_to_db, save_df_to_csv, load_from_db

from .datasets.string_database import PPIData

#############
#   GRAPHS  #
#############
from .graphs import BiGraph

#############
#   EDGES   #
#############

from .edges import (
    map_nodes,
    features_labels,
    features_labels_edges,
    features_labels_idx,
    features_labels_edges_idx,
    split_data,
    split_edge_data,
)

#########################
#   EDGES PREDICTIONS   #
#########################

from .edges import (
    predict,
    prediction_results
)

##################
#   EMPEDDINGS   #
##################
from .embeddings import Node2Vec
from .embeddings import GGVec
from .embeddings import ProNE
from .embeddings import EmbeddingGenerator

#############
#   MODELS  #
#############
from .models import (
    models,
    train_model,
    train_tf_model,
    validate_model
)

from .models import ModelEvaluation

##############
#  __all__   #
##############
__all__ = [
    # Dataset
    "BigQueryClient",
    "GraphQLClient",
    "INDIRECT_SCORES",
    "DIRECT_SCORES",
    "save_to_db",
    "save_df_to_csv",
    "load_from_db",
    "PPIData",
    
    # Graphs
    "BiGraph",
    
    # Edges
    "map_nodes",
    "features_labels",
    "features_labels_edges",
    "features_labels_idx",
    "features_labels_edges_idx",
    "split_data",
    "split_edge_data",
    
    # Edge Predictions
    "predict",
    "prediction_results",
    
    # Embeddings
    "Node2Vec",
    "GGVec",
    "ProNE",
    "EmbeddingGenerator",
    
    # Models
    "models",
    "train_model",
    "train_tf_model",
    "validate_model",
    "ModelEvaluation",
]
