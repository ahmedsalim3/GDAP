from typing import Optional

import networkx as nx
import pandas as pd

from .helper import *

# os.makedirs('assets', exist_ok=True)


class GraphComposer:
    def __init__(self, disease_id, disease_name, genes=None):
        self.disease_id = disease_id
        self.disease_name = disease_name
        self.genes = genes if genes is not None else []
        self.base_url = "https://api.platform.opentargets.org/api/v4/graphql"
        self.query = """
        query DiseaseAssociationsQuery($efoId: String!){
          disease(efoId: $efoId){
            id
            name
            associatedTargets{
              count
              rows{
                target{
                  id
                  approvedSymbol
                }
                score
                datasourceScores {
                  id
                  score
                }
              }
            }
          }
        }
        """
        self.api_res = None
        self.open_target_df = None
        self.features = None
        self.STRING_df = None
        self.open_target_G = None
        self.STRING_G = None
        self.composed_G = None

    def fetch_graphql_data(self):
        self.api_res = fetch_OT_data_from_api(self.base_url, self.query, {"efoId": self.disease_id})

    def process_open_target_data(self):
        if self.api_res is None:
            raise RuntimeError("API response is not available. Run fetch_graphql_data() first.")
        self.open_target_df = process_OT_data(self.api_res)

    def aggregate_data(self):
        if self.open_target_df is None:
            raise RuntimeError("Open target data is not available. Run process_open_target_data() first.")
        self.agg_df, self.gene_symbols, self.features = aggregate_OT_data(self.open_target_df)

    def fetch_string_data(self):
        return fetch_ST_data(self.genes or self.gene_symbols)

    def process_string_data(self):
        data = self.fetch_string_data()
        self.STRING_df = process_ST_data(data)

    def construct_graph(
        self,
        df: pd.DataFrame,
        source_col: str,
        target_col: str,
        edge_attr_col: str,
        plot: bool = False,
        save: bool = False,
        filename: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> nx.Graph:
        return construct_graph(df, source_col, target_col, edge_attr_col, plot, save, filename, figsize)

    def compose_graphs(self):
        if self.open_target_df is not None and self.STRING_df is not None:
            self.open_target_df["datasourceScores_score"] = self.open_target_df["datasourceScores_score"].round(
                3
            )  # round for better plotting
            self.open_target_G = self.construct_graph(
                self.open_target_df, "datasourceScores_id", "approvedSymbol", "datasourceScores_score"
            )
            for u, v, data in self.open_target_G.edges(data=True):
                data["weight"] = data.pop("datasourceScores_score")
                data["type"] = "disease-gene"
            self.STRING_G = self.construct_graph(self.STRING_df, "preferredName_A", "preferredName_B", "score")
            for u, v, data in self.STRING_G.edges(data=True):
                data["weight"] = data.pop("score")
                data["type"] = "protein-protein"
            self.composed_G = nx.compose(self.open_target_G, self.STRING_G)
            return self.composed_G
        else:
            print("DataFrames for Open Target or STRING data are not available, fetch them first")
            return None

    def plot_community_detection(
        self, plot: bool = True, save: bool = False, filename: Optional[str] = None, figsize: tuple = (30, 20)
    ):
        if self.composed_G is None:
            raise RuntimeError("Composed graph is not available. Run compose_graphs() first.")
        plot_community_detection(self.composed_G, "weight", plot, save, filename, figsize)

    def visualize_graphs(
        self, plot: bool = True, save: bool = False, filename: Optional[str] = None, figsize: tuple = (30, 20)
    ):
        if self.composed_G is None:
            raise RuntimeError("Composed graph is not available. Run compose_graphs() first.")
        visualize_graphs(self.composed_G, plot, save, filename, figsize)

    def get_merged_dataframe(self):
        if self.composed_G is None:
            raise RuntimeError("Composed graph is not available. Run compose_graphs() first.")

        edges_data = []
        for u, v, data in self.composed_G.edges(data=True):
            edges_data.append({"source": u, "target": v, "weight": data.get("weight"), "type": data.get("type")})

        merged_df = pd.DataFrame(edges_data)
        return merged_df

    def process_all(self, plot=False, filename=None, save=None):
        self.fetch_graphql_data()
        self.process_open_target_data()
        self.aggregate_data()
        self.process_string_data()
        self.compose_graphs()
        self.get_merged_dataframe()
        self.visualize_graphs(plot=plot, filename=filename, save=save)
        self.plot_community_detection(plot=plot, filename=filename, save=save)
