"""
Module providing a client for interacting with the OpenTargets GraphQL API.

It allows you to:
  1. Fetch the total count of associated targets for a given disease.
  2. Retrieve disease target data from the OpenTargets GraphQL API.
  3. Iterate through the pages of the results to gather all available data for a disease.
  4. Extract and flatten the GraphQL response data into a pandas DataFrame.

Resources:
  - OpenTargets API Schema: https://api.platform.opentargets.org/api/v4/graphql/schema
  - GraphQL API Documentation: https://platform-docs.opentargets.org/data-access/graphql-api
  - GraphQL API Playground: https://api.platform.opentargets.org/api/v4/graphql/browser
  - GraphQL vs BigQuery: https://community.opentargets.org/t/how-to-find-targets-associated-with-a-disease-using-the-new-graphql-api-or-google-bigquery/122
"""

from typing import Any, Dict, List

import pandas as pd
import requests

from .queries import GET_DISEASE_COUNT, GET_DISEASE_TARGETS


class GraphQLClient:
    """Client for interacting with the OpenTargets GraphQL API."""

    def __init__(self) -> None:
        """Initialize the GraphQL client with API configuration."""
        self.api_url = "https://api.platform.opentargets.org/api/v4/graphql"
        self.headers = {
            "Accept-Encoding": "gzip, deflate, br",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive",
            "DNT": "1",
            "Origin": "https://api.platform.opentargets.org",
        }

    def fetch_disease_count(self, efo_id: str) -> int:
        """Fetch the total count of associated targets for the disease."""
        variables = {"efoId": efo_id}
        payload = {"query": GET_DISEASE_COUNT, "variables": variables}
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        return result["data"]["disease"]["associatedTargets"]["count"]

    def fetch_disease(self, efo_id: str, page_index: int, page_size: int) -> Dict[str, Any]:
        """Fetch disease targets data from the GraphQL API and return it as a JSON object."""
        variables = {"efoId": efo_id, "pageSize": page_size, "pageIndex": page_index}
        payload = {"query": GET_DISEASE_TARGETS, "variables": variables}
        response = requests.post(self.api_url, json=payload, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def extract_data(self, result: Dict[str, Any]) -> pd.DataFrame:
        """Extract and flatten the data from the GraphQL response."""
        data_list: List[Dict[str, Any]] = []
        disease_name = result["data"]["disease"]["name"]
        rows = result["data"]["disease"]["associatedTargets"]["rows"]
        for row in rows:
            gene_data = {
                "disease_name": disease_name,
                "symbol": row["target"]["approvedSymbol"],
                "globalScore": row["score"],
            }
            for datasource in row["datasourceScores"]:
                gene_data[datasource["id"]] = datasource["score"]
            data_list.append(gene_data)
        df = pd.DataFrame(data_list)
        return df

    def fetch_full_data(self, efo_id: str) -> pd.DataFrame:
        """Fetch all pages of data for a given disease and combine into a single DataFrame."""
        # set a reasonable page size (API limit is 3000)
        total_targets = self.fetch_disease_count(efo_id)
        page_size = min(3000, total_targets)
        page_number = 0
        data: List[pd.DataFrame] = []

        while True:
            result = self.fetch_disease(efo_id, page_number, page_size)
            df = self.extract_data(result)

            if df.empty:
                break  # stop when no more data is available

            data.append(df)
            page_number += 1

            # break if we've fetched all data
            if len(data) * page_size >= total_targets:
                break
        final_df = pd.concat(data, ignore_index=True)
        final_df = final_df[~final_df["symbol"].duplicated(keep="first")]
        return final_df
