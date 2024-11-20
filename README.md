# Machine Learning - Gene-Disease Association Prediction App

This project is composed of three levels:
- Level 1: develop binary classification model focusing on only one disease (diabetes mellitus or breast cancer)
- Level 2: integrate score/confidence level based on how much and type of evidence from OpenTargets and StringDB.
- Level 3: generalize model to work with multiple diseases.


## Requirements

To install and run the project, you will need:
- **Python 3.x**: Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).
- **pip**: Python package installer should be available.
- **Dependencies**: The project requires the Python packages listed in [requirements.txt](./requirements.txt).

## How to install


1. **Clone the Repository** from terminal:
    ```bash
    git clone https://github.com/mentorchains/BI-ML_Disease-Prediction_2024.git
    cd BI-ML_Disease-Prediction_2024
    ```

2. **Checkout to `ahmed` branch**:
    ```bash
    git checkout ahmed
    git pull origin ahmed
    ```

3. **Create a Virtual Environment** (optional but recommended):
    - On Linux/MacOS:
        ```bash
        python3 -m venv <envname> # <envname> is your environment name
        source <envname>/bin/activate
        ```
    - On Windows:
        ```bash
        python -m venv <envname> # <envname> is your environment name
        .\<envname>\Scripts\activate
        ```
    - Using Conda:
        ```bash
        conda create --name <envname> python=3.12.2 # <envname> is your environment name
        conda activate <envname>
        ```

3. **Install Dependencies:**:
    ```bash
    pip install -r requirements.txt
    ```
## How to Run the Script

1. Go to [Open Target Platform](https://platform.opentargets.org/) and obtain the disease `EFO ID`.

2. Update the configuration in [config.py](./src/config.py) for your experiment. If you choose BigQuery as a data source, ensure you set up your `GOOGLE_APPLICATION_CREDENTIALS` and follow the [steps](./docs/reports/google_cloud_setup.md) to obtain the necessary JSON key files.

3. From the project root, run the script:

    ```bash
    cd src/
    python3 main.py
    ```

## How to Run the App

Running Locally

1. Run the following command from the project root:

    ```sh
    # running as a module
    python3 -m streamlit run app/app.py
    ```
    
2. Alternatively, you can install the package and run it from the app directory:

    ```sh
    pip install .
    streamlit run app/Home.py
    ```

## Running the App via Docker

To run the app using Docker, follow these steps:

1. Build the Docker image:

  ```sh
  docker build -t image_name .
  ```

2. Run the Docker container:

  ```sh
  docker run -p 8501:8501 image_name
  ```

## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
BI-ML_Disease-Prediction_2024
## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
BI-ML_Disease-Prediction_2024
├── CHANGELOG.md            <- Log of changes made
├── README.md
├── .gitignore              <- Specifies intentionally untracked files to ignore by git
├── requirements.txt        <- Python dependencies
├── start_app.sh            <- Streamlit app start script
│
│
├── data                    <- Dir structure for data
│   ├── external            <- Data from third party sources
│   ├── interim             <- Intermediate data that has been transformed
│   ├── processed           <- The final, canonical datasets and results
│   └── raw                 <- The original, immutable data dump
│
├── docs                    <- Dir to store documentation
│   ├── icons
│   ├── images
│   ├── index.md
│   └── reports
│
├── notebooks               <- Jupyter notebooks and experiments
│   ├── main.ipynb
│   └── network-analysis.ipynb
│
├── src                     <- Source code for the project
│   ├── config.py
│   ├── gene_disease        <- Main package
│   └── main.py
│
└── app                     <- Streamlit apps and associated files
    ├── app.py              <- Streamlit app-1
    ├── st_app.py           <- Streamlit app-2 (st_pages)
    ├── models/
    ├── streamlit_helpers/
    ├── tools/
    ├── utils/
    └── views/

```

## Team Structure and Contribution

@[ahmedsalim3](https://github.com/ahmedsalim3)

## References

Provide references of repositories, articles, other work used by your teams.

[DO NOT REMOVE]

Template repo derived from: http://drivendata.github.io/cookiecutter-data-science

Template created by: @samuelbharti
