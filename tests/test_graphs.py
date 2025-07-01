import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import networkx as nx
import tempfile
import os
from gdap.graphs import BiGraph


class TestBiGraph(unittest.TestCase):
    def setUp(self):
        self.ppi_df = pd.DataFrame({
            'GeneName1': ['P1', 'P2', 'P3', 'P4'],
            'GeneName2': ['P2', 'P3', 'P4', 'P1'],
            'combined_score': [0.8, 0.9, 0.7, 0.85]
        })

        self.ot_df = pd.DataFrame({
            'disease_name': ['cardiovascular', 'cardiovascular', 'cardiovascular'],
            'symbol': ['P1', 'P2', 'P5'],
            'score': [0.7, 0.8, 0.6]
        })

    def test_create_graph_basic(self):
        G, pos_edges, neg_edges = BiGraph.create_graph(
            self.ot_df, self.ppi_df, negative_to_positive_ratio=2, output_dir="/tmp"
        )
        self.assertIsInstance(G, nx.Graph)
        self.assertIsInstance(pos_edges, list)
        self.assertIsInstance(neg_edges, list)
        self.assertGreater(len(pos_edges), 0)
        self.assertGreater(len(neg_edges), 0)

    def test_create_graph_with_ratio(self):
        ratios = [1, 2, 5, 10]

        with tempfile.TemporaryDirectory() as temp_dir:
            for ratio in ratios:
                G, pos_edges, neg_edges = BiGraph.create_graph(
                    self.ot_df,
                    self.ppi_df,
                    negative_to_positive_ratio=ratio,
                    output_dir=temp_dir
                )

                if len(pos_edges) > 0:
                    actual_ratio = len(neg_edges) / len(pos_edges)
                    self.assertLessEqual(actual_ratio, ratio + 1)

    def test_create_graph_empty_data(self):
        empty_ot_df = pd.DataFrame(columns=['disease_name', 'symbol', 'score'])
        empty_ppi_df = pd.DataFrame(columns=['GeneName1', 'GeneName2', 'combined_score'])

        G, pos_edges, neg_edges = BiGraph.create_graph(
            empty_ot_df,
            empty_ppi_df,
            negative_to_positive_ratio=2,
            output_dir=None
        )

        self.assertIsInstance(G, nx.Graph)
        self.assertEqual(len(pos_edges), 0)
        self.assertEqual(len(neg_edges), 0)

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_visualize_sample_graph(self, mock_close, mock_savefig):
        G, _, _ = BiGraph.create_graph(
            self.ot_df, self.ppi_df, negative_to_positive_ratio=2, output_dir="/tmp"
        )
        BiGraph.visualize_sample_graph(G, self.ot_df, node_size=300, output_dir="/tmp")
        mock_savefig.assert_called()

    def test_graph_properties(self):
        G, _, _ = BiGraph.create_graph(
            self.ot_df, self.ppi_df, negative_to_positive_ratio=2, output_dir="/tmp"
        )
        self.assertGreater(len(G.nodes()), 0)
        self.assertGreater(len(G.edges()), 0)


if __name__ == "__main__":
    unittest.main()
