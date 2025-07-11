import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from gdap.datasets.open_targets import BigQueryClient, save_to_db, load_from_db


class TestBigqueryFetcher(unittest.TestCase):

    @patch('gdap.datasets.open_targets.bigquery_client.BigQueryClient.init_bq_client')
    def test_init_bq_client(self, mock_bq_client):
        mock_client = MagicMock()
        mock_bq_client.return_value = mock_client
        bigquery = BigQueryClient()
        self.assertIsNotNone(bigquery.client)

    @patch('gdap.datasets.open_targets.bigquery_client.BigQueryClient.init_bq_client')
    @patch('gdap.datasets.open_targets.bigquery_client.BigQueryClient.execute_query')
    def test_execute_query(self, mock_run_query, mock_init):
        mock_client = MagicMock()
        mock_init.return_value = mock_client
        bigquery = BigQueryClient()
        mock_run_query.return_value = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        params = {'disease_id': 'EFO_0000319'}
        df = bigquery.execute_query('SELECT * FROM test_table WHERE diseaseId = "{disease_id}"', params)
        self.assertEqual(len(df), 2)
        self.assertIn('col1', df.columns)


class TestSQLiteDB(unittest.TestCase):

    @patch('sqlite3.connect')
    @patch('pandas.DataFrame.to_sql')
    def test_save_to_db(self, to_sql, connect):
        mock_conn = MagicMock()
        connect.return_value.__enter__.return_value = mock_conn
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        save_to_db(df, 'test_db.db', 'test_table')
        to_sql.assert_called_once_with('test_table', mock_conn, if_exists='replace', index=False)

    @patch('sqlite3.connect')
    @patch('pandas.read_sql')
    def test_load_from_db(self, read_sql, connect):
        mock_conn = MagicMock()
        connect.return_value.__enter__.return_value = mock_conn
        mock_df = pd.DataFrame({'col1': [1], 'col2': [2]})
        read_sql.return_value = mock_df
        df = load_from_db('test_db.db', 'test_table')
        read_sql.assert_called_once_with("SELECT * FROM test_table", mock_conn)
        self.assertTrue(df.equals(mock_df))


if __name__ == '__main__':
    unittest.main()
