import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import networkx as nx
from gdap.embeddings import Node2Vec, ProNE, GGVec, EmbeddingGenerator


class TestNode2Vec(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.node2vec = Node2Vec(n_components=32, walklen=10, verbose=False)

    def test_initialization(self):
        self.assertEqual(self.node2vec.n_components, 32)
        self.assertEqual(self.node2vec.walklen, 10)
        self.assertFalse(self.node2vec.verbose)

    @patch('gdap.embeddings.node2vec.Node2Vec')
    def test_fit_transform(self, mock_node2vec_class):
        mock_model = MagicMock()
        mock_node2vec_class.return_value = mock_model
        mock_embeddings = np.random.rand(4, 32)
        mock_model.fit.return_value = mock_model
        mock_model.transform.return_value = mock_embeddings

        result = self.node2vec.fit_transform(self.graph)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 32))


class TestProNE(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.prone = ProNE(n_components=32, step=5, mu=0.2, theta=0.5, exponent=0.75, verbose=False)

    def test_initialization(self):
        self.assertEqual(self.prone.n_components, 32)
        self.assertEqual(self.prone.step, 5)
        self.assertEqual(self.prone.mu, 0.2)
        self.assertEqual(self.prone.theta, 0.5)
        self.assertEqual(self.prone.exponent, 0.75)
        self.assertFalse(self.prone.verbose)

    @patch('gdap.embeddings.prone.ProNE')
    def test_fit_transform(self, mock_prone_class):
        mock_model = MagicMock()
        mock_prone_class.return_value = mock_model
        mock_embeddings = np.random.rand(4, 32)
        mock_model.fit.return_value = mock_model
        mock_model.transform.return_value = mock_embeddings

        self.prone.fit_transform = MagicMock(return_value=mock_embeddings)

        result = self.prone.fit_transform(self.graph)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 32))


class TestGGVec(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.ggvec = GGVec(n_components=64, order=3, verbose=False)

    def test_initialization(self):
        self.assertEqual(self.ggvec.n_components, 64)
        self.assertEqual(self.ggvec.order, 3)
        self.assertFalse(self.ggvec.verbose)

    @patch('gdap.embeddings.ggvec.GGVec')
    def test_fit_transform(self, mock_ggvec_class):
        mock_model = MagicMock()
        mock_ggvec_class.return_value = mock_model
        mock_embeddings = np.random.rand(4, 64)
        mock_model.fit.return_value = mock_model
        mock_model.transform.return_value = mock_embeddings

        result = self.ggvec.fit_transform(self.graph)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (4, 64))


class TestEmbeddingGenerator(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        self.generator = EmbeddingGenerator()

    def test_simple_node_embedding(self):
        embeddings = self.generator.simple_node_embedding(self.graph, dim=32)
        self.assertIsInstance(embeddings, dict)
        self.assertEqual(len(embeddings), len(self.graph.nodes()))


if __name__ == "__main__":
    unittest.main()
