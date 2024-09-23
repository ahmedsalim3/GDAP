# According to the open target platform's documentation, when using the GraphQL API,
# you need to iterate through each page of the results, running a separate query for each set of 50 results.
# For more information, check out the following resources:
# - [Video Reference](https://youtu.be/_sZR0VxpwqE?si=kJ7pmAg_Uh3PLbU4&t=2384)
# - [Google Presentation](https://docs.google.com/presentation/d/16_99z8Su8j8HbFuhxSlTRZZCW67hNpk0To7UfpoeiTg/edit#slide=id.ged473f4bb5_4_16)
#
# This class provides a more efficient approach:
# 1. It first fetches the total count of associated targets for the given disease.
# 2. Then it iterates through the pages continuously until all data is fetched.
#
# API Documentation:
# - Schema: https://api.platform.opentargets.org/api/v4/graphql/schema
# - GraphQL API Documentation: https://platform-docs.opentargets.org/data-access/graphql-api
# - GraphQL API Playground: https://api.platform.opentargets.org/api/v4/graphql/browser
#
# GraphQL API vs Google BigQuery:
# Link: https://community.opentargets.org/t/how-to-find-targets-associated-with-a-disease-using-the-new-graphql-api-or-google-bigquery/122

import requests
import pandas as pd
import os
from src.open_targets.utils import save_df_to_csv, save_to_db, load_from_db

class GraphQLClient:
    def __init__(self):
        self.api_url = 'https://api.platform.opentargets.org/api/v4/graphql'
        self.headers = {
            'Accept-Encoding': 'gzip, deflate, br',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Origin': 'https://api.platform.opentargets.org'
        }
        
    def fetch_disease_count(self, efo_id):
        """Fetch the total count of associated targets for the disease."""
        query = """
        query GetDiseaseTargetCount($efoId: String!) {
          disease(efoId: $efoId) {
            associatedTargets {
              count
            }
          }
        }
        """
        variables = {'efoId': efo_id}
        payload = {'query': query, 'variables': variables}
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        return result['data']['disease']['associatedTargets']['count']
    
    def fetch_disease(self, efo_id, page_index, page_size):
        """Fetch disease targets data from the GraphQL API and return it as a JSON object."""
        query = """
        query GetDiseaseTargets(
          $efoId: String!,
          $pageSize: Int!,
          $pageIndex: Int!
        ) {
          disease(efoId: $efoId) {
            id
            name
            associatedTargets(page: { size: $pageSize, index: $pageIndex }) {
              count
              rows {
                target {
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
        variables = {
            'efoId': efo_id,
            'pageSize': page_size,
            'pageIndex': page_index
        }
        payload = {'query': query, 'variables': variables}
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def extract_data(self, result):
        """Extract and flatten the data from the GraphQL response."""
        data_list = []
        disease_name = result['data']['disease']['name']
        rows = result['data']['disease']['associatedTargets']['rows']
        for row in rows:
            gene_data = {
              'disease_name': disease_name,
              'symbol': row['target']['approvedSymbol'],
              'globalScore': row['score']
            }
            for datasource in row['datasourceScores']:
                gene_data[datasource['id']] = datasource['score']
            data_list.append(gene_data)
        df = pd.DataFrame(data_list)
        return df
      
    def fetch_full_data(self, efo_id):
        """Fetch all pages of data for a given disease and combine into a single DataFrame."""
        page_size = self.fetch_disease_count(efo_id)
        page_number = 0
        data = []
        
        while True:
            result = self.fetch_disease(efo_id, page_number, page_size)
            df = self.extract_data(result)
            
            if df.empty:
                break  # Stop when no more data is available
            
            data.append(df)
            page_number += 1
        
        final_df = pd.concat(data, ignore_index=True)
        final_df = final_df[~final_df['symbol'].duplicated(keep='first')]
        return final_df
      

# # Example usage - Fetch data for one page
# if __name__ == "__main__":
#     table_name = 'global_score'
#     disease_id = 'EFO_0000319'
#     page_number = 23
#     page_size = 50
    
#     try:
#         # Initialize GraphQL Client
#         graphql_client = GraphQLClient()
        
#         # Fetch disease targets data
#         result = graphql_client.fetch_disease(disease_id, page_number, page_size)     
#         df = graphql_client.extract_data(result)
#         disease_name = df.disease_name.iloc[0].split()[0]
        
#         # Create save folder path
#         folder = os.path.join('data', disease_name)
#         os.makedirs(folder, exist_ok=True)
        
#         # Save DataFrame to database
#         db_path = os.path.join(folder, f'GraphQL_{disease_name}_page-{page_number}.db')
#         save_to_db(df, db_path, table_name)
        
#         # Load from database and save to CSV
#         df_loaded = load_from_db(db_path, table_name)
#         csv_path = os.path.join(folder, f'GraphQL_{disease_name}_page-{page_number}.csv')
#         save_df_to_csv(df_loaded, csv_path)
        
#     except Exception as e:
#         print(f"An error occurred: {e}")
        

# Example usage - Fetch full data
if __name__ == "__main__":
    table_name = 'global_score'
    disease_id = 'EFO_0000319'
    
    try:
        # Initialize GraphQL Client
        graphql_client = GraphQLClient()
        
        # Fetch full data for the disease
        df = graphql_client.fetch_full_data(disease_id)
        disease_name = df.disease_name.iloc[0].split()[0]
        
        # Create save folder path
        folder = os.path.join('data', disease_name)
        os.makedirs(folder, exist_ok=True)
        
        # Save DataFrame to database
        db_path = os.path.join(folder, f'GraphQL_{disease_name}.db')
        save_to_db(df, db_path, table_name)
        
        # Load from database and save to CSV
        df_loaded = load_from_db(db_path, table_name)
        csv_path = os.path.join(folder, f'GraphQL_{disease_name}.csv')
        save_df_to_csv(df_loaded, csv_path)
                
    except Exception as e:
        print(f"An error occurred: {e}")
