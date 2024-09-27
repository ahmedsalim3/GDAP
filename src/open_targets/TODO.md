## Google Cloud Setup
- [x] Create a service account in Google Cloud with the necessary roles (BigQuery User)
    - Go to the [Google Cloud Console](https://console.cloud.google.com/welcome?project=stemaway).
    - Select your project or create a new one if needed.
    - In the navgation menue, go to IAM & Admin > Service Accounts.
    - Create a new Service Account or use an existing one.
    - Grant the necessary roles (e.g., BigQuery User) to this service account ID.
    - Create a key for this service account in JSON format.
- [x] Download the service account JSON key.
    - [ ] _If you are testing it locally, you can define the environment variable GOOGLE_APPLICATION_CREDENTIALS to the path of the JSON key file. You can set this in your terminal or directly in your code._
        ```python
        import os
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_json_path_goes_here.json"
        ```

- [x] Set the GOOGLE_APPLICATION_CREDENTIALS environment variable through terminal.
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS='your_json_path_goes_here.json'
    ```
- [x] Verify Google Cloud SDK is installed and configured.
    ```bash
    echo $GOOGLE_APPLICATION_CREDENTIALS  # Verify it
    unset GOOGLE_APPLICATION_CREDENTIALS  # Unset it
    ```

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