import os
from open_targets import BigQueryClient, DIRECT_SCORES, INDIRECT_SCORES
from open_targets import save_df_to_csv, save_to_db, load_from_db
from open_targets import GraphQLClient


def bigquery_fetcher(disease_id, credentials_path=None):
    params = {"disease_id": disease_id}
    table_names = ["indirect_scores", "direct_scores"]

    # Initialize BigQuery client
    bq_client = BigQueryClient(credentials_path=credentials_path)

    # Fetch data
    df_indirect = bq_client.execute_query(INDIRECT_SCORES, params)
    df_direct = bq_client.execute_query(DIRECT_SCORES, params)

    # Save data to database
    disease_name = df_indirect.disease_name.iloc[0].split()[0]
    folder = os.path.join("data", disease_name)
    os.makedirs(folder, exist_ok=True)
    db_path = os.path.join(folder, f"{disease_name}.db")

    save_to_db(df_indirect, db_path, table_names[0])
    save_to_db(df_direct, db_path, table_names[1])

    # Save to CSV
    for table, df in zip(table_names, [df_indirect, df_direct]):
        df_loaded = load_from_db(db_path, table)
        csv_path = os.path.join(folder, f"{disease_name}_{table}.csv")
        save_df_to_csv(df_loaded, csv_path)


def graphql_fetcher(disease_id):
    table_name = "global_score"

    # Initialize GraphQL Client
    gql_client = GraphQLClient()

    # Fetch data
    df = gql_client.fetch_full_data(disease_id)
    disease_name = df.disease_name.iloc[0].split()[0]

    # Save data to database
    folder = os.path.join("data", disease_name)
    os.makedirs(folder, exist_ok=True)
    db_path = os.path.join(folder, f"GraphQL_{disease_name}.db")
    save_to_db(df, db_path, table_name)

    # Save to CSV
    df_loaded = load_from_db(db_path, table_name)
    csv_path = os.path.join(folder, f"GraphQL_{disease_name}.csv")
    save_df_to_csv(df_loaded, csv_path)


def graphql_page_fetcher(disease_id, page_number, page_size=50):
    table_name = "global_score"

    # Initialize GraphQL Client
    gql_client = GraphQLClient()

    # Fetch data
    result = gql_client.fetch_disease(disease_id, page_number, page_size)
    df = gql_client.extract_data(result)
    disease_name = df.disease_name.iloc[0].split()[0]

    # Save data to database
    folder = os.path.join("data", disease_name)
    os.makedirs(folder, exist_ok=True)
    db_path = os.path.join(folder, f"GraphQL_{disease_name}_page-{page_number}.db")
    save_to_db(df, db_path, table_name)

    # Save to CSV
    df_loaded = load_from_db(db_path, table_name)
    csv_path = os.path.join(folder, f"GraphQL_{disease_name}_page-{page_number}.csv")
    save_df_to_csv(df_loaded, csv_path)


if __name__ == "__main__":
    disease_id = "EFO_0000319"
    bigquery_fetcher(disease_id)
    graphql_fetcher(disease_id)
    graphql_page_fetcher(disease_id, page_number=1, page_size=50)
