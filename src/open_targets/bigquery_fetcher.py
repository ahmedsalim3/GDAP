# Alternatively, You can these SQL queries in BigQuery for your specific disease
# and download the results as CSV/JSON:
# Link: https://console.cloud.google.com/bigquery?sq=119305062578:ae33e769ecec43ceb055565f7d6c74df

# The database tables are available here:
# Link: https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.06/output/etl/json/

# More info can be found on the community:
# Link: https://community.opentargets.org/t/returning-all-associations-data-using-the-platform-api/324/2

# GraphQL API vs Google BigQuery:
# Link: https://community.opentargets.org/t/how-to-find-targets-associated-with-a-disease-using-the-new-graphql-api-or-google-bigquery/122

indirect_scores = """
SELECT
  overall_indirect.diseaseId AS disease_id,
  diseases.name AS disease_name,
  targets.approvedSymbol AS symbol,
  overall_indirect.score AS indirect_score,
  overall_indirect.evidenceCount AS indirect_evidence_count
FROM
  `open-targets-prod.platform.associationByOverallIndirect` AS overall_indirect
JOIN
  `open-targets-prod.platform.diseases` AS diseases
ON
  overall_indirect.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.targets` AS targets
ON
  overall_indirect.targetId = targets.id
WHERE
  overall_indirect.diseaseId = '{disease_id}'
ORDER BY
  indirect_score DESC;
"""

direct_scores = """
SELECT
  overall_direct.diseaseId AS disease_id,
  diseases.name AS disease_name,
  targets.approvedSymbol AS symbol,
  overall_direct.score AS direct_score,
  overall_direct.evidenceCount AS direct_evidence_count
FROM
  `open-targets-prod.platform.associationByOverallDirect` AS overall_direct
JOIN
  `open-targets-prod.platform.diseases` AS diseases
ON
  overall_direct.diseaseId = diseases.id
JOIN
  `open-targets-prod.platform.targets` AS targets
ON
  overall_direct.targetId = targets.id
WHERE
  overall_direct.diseaseId = '{disease_id}'
ORDER BY
  direct_score DESC;
"""

from google.cloud import bigquery
import os
from src.open_targets.utils import save_df_to_csv, save_to_db, load_from_db
from google.oauth2 import service_account
import streamlit as st


class BigQueryClient:
    def __init__(self, deploy=False):
        if deploy:
            self.client = self.init_bq_client_deploy()
        else:
            self.client = self.init_bq_client()

    def init_bq_client(self):
        """
        Initializes a BigQuery client using credentials from the environment variable
        'GOOGLE_APPLICATION_CREDENTIALS'.

        The environment variable should be set to the path of your service account JSON key.

        For details on how to set up your credentials, refer to the `TODO.md` file.
        """
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError(
                "The environment variable GOOGLE_APPLICATION_CREDENTIALS is not set."
            )
        client = bigquery.Client.from_service_account_json(credentials_path)
        # # uncomment those for local use case:
        # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "stemaway-d5a18133ff83.json"
        # client = bigquery.Client()
        return client

    def init_bq_client_deploy(self):
        """Initializes a BigQuery client using credentials from Streamlit secrets for deployment."""
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"]
        )
        print(f'Credentials {"Found" if credentials is not None else "Not Found"}')
        client = bigquery.Client(credentials=credentials)
        print(f"Client initialized for project: {client.project}")
        return client

    def execute_query(self, query, params=None):
        """Run a SQL query on BigQuery and return the results as a DataFrame."""
        if params:
            query = query.format(**params)  # Format query with parameters
        query_job = self.client.query(query)  # API request
        res = query_job.result()
        df = res.to_dataframe()
        df = df[~df["symbol"].duplicated(keep="first")]
        return df


# Example usage
if __name__ == "__main__":
    params = {"disease_id": "EFO_0000319"}
    table_name = ["indirect_scores", "direct_scores"]
    try:
        # Initialize BigQuery client
        bq_client = BigQueryClient()

        # Fetch indirect and direct scores data
        df_indirect = bq_client.execute_query(indirect_scores, params)
        df_direct = bq_client.execute_query(direct_scores, params)

        # Create save folder path
        disease_name = df_indirect.disease_name.iloc[0].split()[0]
        folder = os.path.join("data", disease_name)
        os.makedirs(folder, exist_ok=True)

        # Save df to database in two separated tables
        db_path = os.path.join(folder, f"{disease_name}.db")
        save_to_db(df_indirect, db_path, table_name[0])
        save_to_db(df_direct, db_path, table_name[1])

        # Load from database and save to CSV
        # Table 1: indirect_scores
        df_loaded = load_from_db(db_path, table_name[0])
        csv_path = os.path.join(folder, f"{disease_name}_indirect_scores.csv")
        save_df_to_csv(df_loaded, csv_path)
        # Table 2: direct_scores
        df_loaded = load_from_db(db_path, table_name[1])
        csv_path = os.path.join(folder, f"{disease_name}_direct_scores.csv")
        save_df_to_csv(df_loaded, csv_path)

    except Exception as e:
        print(f"An error occurred: {e}")
