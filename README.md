# Machine Learning - Gene-Disease Association Prediction App

This project is composed of three levels:
- Level 1: develop binary classification model focusing on only one disease (diabetes mellitus or breast cancer)
- Level 2: integrate score/confidence level based on how much and type of evidence from OpenTargets and StringDB.
- Level 3: generalize model to work with multiple diseases.


## Requirements

To install and run the project, you will need:
- **Python 3.x**: Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).
- **pip**: Python package installer should be available.
- **Dependencies**: The project requires the Python packages listed in [requirements.txt](./configs/requirements.txt) or [conda_requirements.txt](./configs/conda_requirements.txt).

## How to install


1. **Clone the Repository** from terminal:
    ```bash
    git clone https://github.com/mentorchains/BI-ML_Disease-Prediction_2024.git
    cd BI-ML_Disease-Prediction_2024
    ```
2. **Create a Virtual Environment** (optional but recommended):
    - On Linux/MacOS:
        ```bash
        python -m venv venv
        source venv/bin/activate
        ```
    - On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3. **Install Dependencies:**:
    ```bash
    pip install -r requirements.txt
    ```
 4. **If you are using Conda:**

    ```bash
    conda create --name <env> --file conda_requirements.txt  # <env> is your environment name
    ```
## How to Run the Script

1. Go to [Open Target Platform](https://platform.opentargets.org/) and obtain the disease `EFO ID`.

2. Update the configuration in [config.py](./src/config.py) for your experiment. If you choose BigQuery as a data source, ensure you set up your `GOOGLE_APPLICATION_CREDENTIALS` and follow the steps in [TODO](./src/open_targets/TODO.md) to obtain the necessary JSON key files.

3. From the project root, run the script:

    ```bash
    python -m src.main
    ```
4. The results will be saved in the [results/<disease-name>](./results/) directory.

## How to Run the App

1. Run the following command from the project root:
    ```sh
    python -m streamlit run app/Home.py
    ```
2. Open your browser and navigate to your local or network URL:
    - Local URL: `http://localhost:8501`
    - Network URL: `http://192.168.45.100:8501`

## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
BI-ML_Disease-Prediction_2024
├── CHANGELOG.md      <- Log of changes made
│
├── README.md
│
├── .gitignore        <- Specifies intentionally untracked files to ignore by git
│
├── configs           <- Dir to store config files. Conda env, requirements.txt, etc.
│   ├── requirements.txt
│   └── conda_requirements.txt
│
├── data              <- Dir structure.
│   ├── external      <- Data from third party sources
│   │── interim       <- Intermediate data that has been transformed.
│   ├── processed     <- The final, canonical datasets and results
│   └── raw           <- The original, immutable data dump
│
├── docs              <- Dir to store documentation.
│
├── models            <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks         <- Jupyter notebooks and experiments
│   ├── main.ipynb
│   └── network-analysis.ipynb
│
├── src               <- Source code for the project
│   ├── open_targets/
│   ├── graph_analysis/
│   ├── config.py
│   ├── main.py
│   ├── edge_utils.py
│   ├── embeddings.py
│   ├── edge_predictions.py
│   ├── ml_models.py
│   ├── model_evaluation.py
│   ├── bigraph.py
│   └── ppi_data.py
│ 
└── app               <- Streamlit app
    ├── pages/
    ├── screenshots/
    ├── Home.py
    ├── functions.py
    ├── model_parms.py
    ├── model_training.py
    ├── ui.py
    ├── utils.py
    └── visualizations.py

```

## Team Structure and Contribution

Use this space to write your team names and their contribution.

## References

Provide references of repositories, articles, other work used by your teams.

[DO NOT REMOVE]

Template repo derived from: http://drivendata.github.io/cookiecutter-data-science

Template created by: @samuelbharti
