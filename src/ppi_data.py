import pandas as pd

class PPIData:
    """
    This class downloads the PPI dataset from a shared Google Drive link, cleans the data by removing
    missing values, and scales the 'experimental' column. It also limits the number of interactions
    based on the provided 'max_ppi_interactions' value
    """
    def __init__(self, max_ppi_interactions=None):
        """
        max_ppi_interactions : int, optional (default=5,000,000)
            Maximum number of PPI interactions to retain.
        """
        self.max_ppi_interactions = max_ppi_interactions
        
    @staticmethod
    def fetch_string_db_ppi():
        url = 'https://drive.google.com/file/d/1xAR1NNa4fporQjKLkjPwvTjBpR9JiSIV/view?usp=sharing'
        ppi_path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]
        ppi_df = pd.read_csv(ppi_path)
        return ppi_df
    
    def process_ppi_data(self):
        ppi_df = self.fetch_string_db_ppi()
        
        # Clean the data
        ppi_df.dropna(subset=['GeneName1', 'GeneName2'])
        ppi_df = ppi_df[(ppi_df['GeneName1'] != '') & (ppi_df['GeneName2'] != '')]
        
        # Normalizing
        ppi_df['experimental'] = ppi_df['experimental'] / 1000
        
        # Limit the number of interactions if needed
        if self.max_ppi_interactions is not None:
            if len(ppi_df) > self.max_ppi_interactions:
                ppi_df = ppi_df.sample(n=self.max_ppi_interactions, random_state=42)
        
        # Rename the 'experimental' column to 'combined_score'
        ppi_df.rename(columns={'experimental': 'combined_score'}, inplace=True)
        ppi_df.sort_values(by='combined_score', ascending=False, inplace=True)
        ppi_df.reset_index(drop=True, inplace=True)
        
        print(f"Final PPI data shape: {ppi_df.shape}")
        return ppi_df
