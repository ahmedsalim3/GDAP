"""
This class is intended to fetch open-target diseases using the BigQuery approach.
Two SQL queries can be executed against the BigQuery Database: DIRECT_SCORES and INDIRECT_SCORES.

The database tables are available here:
Link: https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.06/output/etl/json/

More info can be found on the community:
Link: https://community.opentargets.org/t/returning-all-associations-data-using-the-platform-api/324/2

GraphQL API vs Google BigQuery:
Link: https://community.opentargets.org/t/how-to-find-targets-associated-with-a-disease-using-the-new-graphql-api-or-google-bigquery/122

NOTE: alternatively, you can use these SQL queries in BigQuery console for your specific disease
      and download the dataset as CSV/JSON:
      Link: https://console.cloud.google.com/bigquery?sq=119305062578:ae33e769ecec43ceb055565f7d6c74df
"""

from google.cloud import bigquery
import os
from google.oauth2 import service_account
import streamlit as st


class BigQueryClient:
    def __init__(self, deploy=False, credentials_path=None):
        """
        Initializes the BigQuery client either for deployment (via Streamlit secrets) 
        or for local development (via environment variables or direct credentials file).
        """
        self.client = self.init_bq_client(deploy, credentials_path)

    def init_bq_client(self, deploy, credentials_path):
        """
        Initializes a BigQuery client using credentials from either a service account or environment variables.
        Handles both local development and deployment configurations.
        """
        if deploy:
            return self.init_bq_client_deploy()
        else:
            return self.init_bq_client_local(credentials_path)

    def init_bq_client_local(self, credentials_path):
        """
        Initializes a BigQuery client using credentials from local environment or a given credentials file.
            1. `credentials_path` (if provided)
            2. Directly from the environment variable GOOGLE_APPLICATION_CREDENTIALS, which is set using export
            1. From the .env file loaded with load_dotenv()
        
        The environment variable should be set to the path of your service account JSON key.
        For details on how to set up your credentials, refer to the TODO.md file.
        """
        
        credentials_path = credentials_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if credentials_path:
            client = bigquery.Client.from_service_account_json(credentials_path)
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            client = bigquery.Client()
        else:
            raise ValueError("The environment variable GOOGLE_APPLICATION_CREDENTIALS is not set.")

        return client

    def init_bq_client_deploy(self):
        """Initializes a BigQuery client using credentials from Streamlit secrets for deployment."""
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        print(f"Credentials {'Found' if credentials else 'Not Found'}")
        client = bigquery.Client(credentials=credentials)
        print(f"Client initialized for project: {client.project}")
        return client

    def execute_query(self, query, params=None):
        """Run a SQL query on BigQuery and returns the results as a DataFrame."""
        if params:
            query = query.format(**params)  # Format query with parameters
        query_job = self.client.query(query)  # API request
        res = query_job.result()
        df = res.to_dataframe()
        df = df[~df["symbol"].duplicated(keep="first")]  # Remove duplicate symbols
        return df
