import os
from config import Config
# ==========================
#       MAIN CLASS
# ==========================
class DiseaseGene:
    def __init__(self, config: Config):
        self.config = config
        self.disease_name = None
        self.G = None
        self.pos_edges = None
        self.neg_edges = None
        self.embeddings = None
        self.node_to_index = None
        self.model = None

    def load_data(self):
        """Load and process datasets (PPI and disease) based on configuration."""
        from disease_gene.datasets.open_targets import BigQueryClient, GraphQLClient, DIRECT_SCORES, INDIRECT_SCORES
        from disease_gene.datasets.string_database import PPIData

        # Load PPI data
        ppi_data = PPIData(max_ppi_interactions=self.config.PPI_INTERACTIONS)
        ppi_df = ppi_data.process_ppi_data()

        # Load disease data based on data source
        if self.config.DATA_SOURCE == "GraphQLClient":
            graphql_client = GraphQLClient()
            ot_df = graphql_client.fetch_full_data(self.config.DISEASE_ID)
        elif self.config.DATA_SOURCE == "BigQueryClient":
            bq_client = BigQueryClient()

            query = self.config.QUERY  # Should be set to either DIRECT_SCORES or INDIRECT_SCORES
            if query not in [DIRECT_SCORES, INDIRECT_SCORES]:
                raise ValueError(f"Invalid query selected: {query}. It must be either DIRECT_SCORES or INDIRECT_SCORES.")
            params = {"disease_id": self.config.DISEASE_ID}
            ot_df = bq_client.execute_query(query, params)

        self.disease_name = ot_df.disease_name.iloc[0].split()[0]
        return ppi_df, ot_df

    def create_graph(self, ppi_df, ot_df):
        """Create graph using PPI data and open-targets data."""
        from disease_gene.graphs import BiGraph
        G, pos_edges, neg_edges = BiGraph.create_graph(
            ot_df, ppi_df, negative_to_positive_ratio=self.config.NEGATIVE_TO_POSITIVE_RATIO, output_dir=self.config.OUTPUT_DIR)
        BiGraph.visualize_sample_graph(G, ot_df, node_size=300, output_dir=self.config.OUTPUT_DIR)
        self.G, self.pos_edges, self.neg_edges = G, pos_edges, neg_edges

    def generate_embeddings(self):
        """Generate node embeddings for the graph."""
        from disease_gene.embeddings import Node2Vec, ProNE, GGVec, EmbeddingGenerator

        save_path = os.path.join(self.config.OUTPUT_DIR, self.disease_name, "embedding_wheel/")
        os.makedirs(save_path, exist_ok=True)

        if self.config.EMBEDDING_MODE == "degree_avg":
            self.embeddings = EmbeddingGenerator.simple_node_embedding(self.G, dim=64)
        elif self.config.EMBEDDING_MODE == "Node2Vec":
            model = Node2Vec(n_components=32, walklen=10, verbose=False)
            self.embeddings = model.fit_transform(self.G)
            model.save(os.path.join(save_path, "n2v_model"))
            model.save_vectors(os.path.join(save_path, "n2v_wheel_model.bin"))
        elif self.config.EMBEDDING_MODE == "ProNE":
            model = ProNE(n_components=32, step=5, mu=0.2, theta=0.5, exponent=0.75, verbose=False)
            self.embeddings = model.fit_transform(self.G)
            model.save(os.path.join(save_path, "prone_model"))
            ProNE.save_vectors(model, os.path.join(save_path, "prone_wheel_model.bin"))
        elif self.config.EMBEDDING_MODE == "GGVec":
            model = GGVec(n_components=64, order=3, verbose=False)
            self.embeddings = model.fit_transform(self.G)
            model.save(os.path.join(save_path, "ggvec_model"))
            GGVec.save_vectors(model, os.path.join(save_path, "ggvec_wheel_model.bin"))

    def extract_features_labels(self, test_size):
        """Extract features, labels, and split data."""
        from disease_gene.edges.edge_utils import features_labels_edges_idx, features_labels_edges, split_edge_data

        # Map nodes to index for embeddings
        from disease_gene.edges.edge_utils import map_nodes
        self.node_to_index = map_nodes(self.G)

        if self.config.EMBEDDING_MODE == "degree_avg":
            X, y, edges = features_labels_edges(self.pos_edges, self.neg_edges, self.embeddings)
        else:
            X, y, edges = features_labels_edges_idx(self.pos_edges, self.neg_edges, self.embeddings, self.node_to_index)

        return split_edge_data(X, y, edges, test_size)

    def train_and_evaluate_model(self, X_train, y_train, X_test, y_test, X_val, y_val):
        """Train and evaluate classification model."""
        from disease_gene.models import sklearn_models, train_model, validate_model
        import joblib

        model, _ = train_model(sklearn_models[self.config.MODEL_NAME], X_train, y_train, model_name=self.config.MODEL_NAME)
        test_results, val_results = validate_model(model, X_test, y_test, X_val, y_val, threshold=0.5)

        print(f"\n{self.config.MODEL_NAME} (Test Set):")
        for metric, value in test_results.items():
            print(f"{metric}: {value:.4f}")

        print(f"\n{self.config.MODEL_NAME} (Validation Set):")
        for metric, value in val_results.items():
            print(f"{metric}: {value:.4f}")

        # Save model
        models_path = os.path.join(self.config.OUTPUT_DIR, self.disease_name, "classifier_models/")
        os.makedirs(models_path, exist_ok=True)
        joblib.dump(model, os.path.join(models_path, f"{self.config.MODEL_NAME}.pkl"))

        return model, models_path

    def plot_model_evaluation(self, model, X_val, y_val, models_path):
        """Plot model evaluation results."""
        from disease_gene.models import ModelEvaluation
        Evaluation = ModelEvaluation(model, X_val, y_val, threshold=0.5, model_name=self.config.MODEL_NAME, figsize=(14, 12), output_dir=models_path)
        Evaluation.plot_evaluation()

    def predict_and_save_results(self, model, X_val, edges_val, models_path, threshold=0.5):
        """Make predictions and save results."""
        from disease_gene.edges.edge_predictions import predict, prediction_results
        associated_proteins, non_associated_proteins = predict(model, X_val, edges_val, threshold=threshold)
        associated_df, non_associated_df = prediction_results(associated_proteins, non_associated_proteins, output_dir=self.config.OUTPUT_DIR)
        return associated_df, non_associated_df


# ==========================
#       MAIN EXECUTION
# # ==========================
if __name__ == "__main__":
    
    # initialize the config
    config = Config()

    # pipeline instance
    pipeline = DiseaseGene(config)

    # load datasets
    ppi_df, ot_df = pipeline.load_data()

    # create graph
    pipeline.create_graph(ppi_df, ot_df)

    # generate embeddings
    pipeline.generate_embeddings()

    # extract features and labels
    X_train, y_train, X_val, y_val, X_test, y_test, edges_train, edges_val, edges_test = pipeline.extract_features_labels(config.TEST_SIZE)

    # train and evaluate model
    model, models_path = pipeline.train_and_evaluate_model(X_train, y_train, X_test, y_test, X_val, y_val)

    # plot evaluation results
    pipeline.plot_model_evaluation(model, X_val, y_val, models_path)

    # make predictions and save results
    associated_df, non_associated_df = pipeline.predict_and_save_results(model, X_val, edges_val, models_path)
