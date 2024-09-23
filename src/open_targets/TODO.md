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
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "stemaway-d5a18133ff83.json" # "your_json_path_goes_here.json"
        ```

- [x] Set the GOOGLE_APPLICATION_CREDENTIALS environment variable through terminal (For deployment).
    ```bash
    export GOOGLE_APPLICATION_CREDENTIALS="stemaway-d5a18133ff83.json"
    export GOOGLE_APPLICATION_CREDENTIALS='/src/open_targets/stemaway-d5a18133ff83.json' # "your_json_path_goes_here.json"
    ```
- [x] Verify Google Cloud SDK is installed and configured.
    ```bash
    echo $GOOGLE_APPLICATION_CREDENTIALS  # Verify it
    unset GOOGLE_APPLICATION_CREDENTIALS  # Unset it
    ```
