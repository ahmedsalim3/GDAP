## Google Cloud Setup
1. Create a service account in Google Cloud with the necessary roles (BigQuery User)
    - Go to the [Google Cloud Console](https://console.cloud.google.com/welcome?project=stemaway).
    - Select your project or create a new one if needed.
    - In the navgation menue, go to IAM & Admin > Service Accounts.
    - Create a new Service Account or use an existing one.
    - Grant the necessary roles (e.g., BigQuery User) to this service account ID.
    - Create a key for this service account in JSON format.

2. Download the service account JSON key.
    - After creating the service account and generating the key, download the JSON key file to your local machine This file will be used for authentication.

3. Set Up Google Cloud Credentials Locally (using `.env` or `direct path`)

    - [ ] **Option 1: Using `.env` File**
   
        - Create a `.env` file in your project’s root directory (if it doesn't already exist).
        - Add the **full path** to your service account JSON key in the .env file:

            ```sh
            GOOGLE_APPLICATION_CREDENTIALS=/home/ahmedsalim/projects/BI-ML_Disease-Prediction_2024/configs/stemaway-d5a18133ff83.json
            ```
        - Install the `python-dotenv` package if it isn't already installed. You can do this by running:

            ```sh
            pip install python-dotenv
            ```
    - [ ] **Option 2: Using the environment variable GOOGLE_APPLICATION_CREDENTIALS**
   
        - You can set this in your terminal or provide the the [`credentials_path`](./bigquery_client.py#L26) directly when initializeing the client
            - [ ] To set the GOOGLE_APPLICATION_CREDENTIALS environment variable through terminal, run these

                ```sh
                export GOOGLE_APPLICATION_CREDENTIALS='your_json_path_goes_here.json'
                ```
                
                ```sh
                echo $GOOGLE_APPLICATION_CREDENTIALS  # Verify it
                unset GOOGLE_APPLICATION_CREDENTIALS  # Unset it
                ```
            
            2. Provide 
        
4. Run the [main file](../main.py) to fetch the desired disease dataset

## Deployment

- [x] Add the key file to your local app secrets `.streamlit/secrets.toml`

    ```toml
    # .streamlit/secrets.toml

    [gcp_service_account]
    type = "service_account"
    project_id = "xxx"
    private_key_id = "xxx"
    private_key = "xxx"
    client_email = "xxx"
    client_id = "xxx"
    auth_uri = "https://accounts.google.com/o/oauth2/auth"
    token_uri = "https://oauth2.googleapis.com/token"
    auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
    client_x509_cert_url = "xxx"
    ```
- [x] Copy your app secrets to the cloud
    Go to the [app dashboard](https://share.streamlit.io/) and in the app's dropdown menu, click on Edit Secrets. Copy the content of secrets.toml into the text area.

## References
- [Connect Streamlit to Google BigQuery](https://docs.streamlit.io/knowledge-base/tutorials/databases/bigquery)