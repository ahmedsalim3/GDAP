import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

app_path = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_path))


@pytest.fixture
def sample_config():
    from src.config import Config
    return Config()


@pytest.fixture
def sample_ppi_data():
    import pandas as pd
    return pd.DataFrame({
        'protein1': ['P1', 'P2', 'P3', 'P4'],
        'protein2': ['P2', 'P3', 'P4', 'P1'],
        'score': [0.8, 0.9, 0.7, 0.85]
    })


@pytest.fixture
def sample_ot_data():
    import pandas as pd
    return pd.DataFrame({
        'disease_name': ['Cardiovascular Disease'],
        'target_id': ['P1', 'P2', 'P5'],
        'score': [0.7, 0.8, 0.6]
    })


@pytest.fixture
def sample_graph():
    import networkx as nx
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G


@pytest.fixture
def sample_embeddings():
    import numpy as np
    return np.random.rand(10, 64)


@pytest.fixture
def sample_model_data():
    import numpy as np
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_test = np.random.rand(20, 10)
    y_test = np.random.randint(0, 2, 20)
    X_val = np.random.rand(20, 10)
    y_val = np.random.randint(0, 2, 20)

    return X_train, y_train, X_test, y_test, X_val, y_val


@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "test_output"
