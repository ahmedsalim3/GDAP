# Machine Learning - Gene-Disease Association Prediction

This project is composed of three levels:
- Level 1: develop binary classification model focusing on only one disease (diabetes mellitus or breast cancer)
- Level 2: integrate score/confidence level based on how much and type of evidence from OpenTargets and StringDB.
- Level 3: generalize model to work with multiple diseases.


## Requirements

To install and run the project, you will need:
- **Python 3.x**: Ensure you have Python 3.x installed. You can download it from [python.org](https://www.python.org/).
- **pip**: Python package installer should be available.
- **Dependencies**: The project requires the Python packages listed in [requirements.txt](requirements.txt).

## How to install


1. **Clone the Repository** from terminal:
    ```bash
    git clone https://github.com/mentorchains/BI-ML_Disease-Prediction_2024.git
    cd BI-ML_Disease-Prediction_2024
    ```
2. **Create a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use '.\venv\Scripts\activate'
    ```
3. **Install Dependencies:**:
    ```bash
    pip install -r requirements.txt
    ```

## How to run

1. Navigate to [main.py](src/main.py) Directory:
    ```bash
    cd src
    ```
2. Run the Script:

    You can execute the script to visualize a composed graph with communities for any disease from the [Open Target Platform](https://platform.opentargets.org/), You will need the `EFO ID` for the disease, and optionally, a Disease Name for better visualization. The script will fetch data from the Open Target Platform and [STRING Database](https://string-db.org/) to generate PPI Composed Graphs.

    To use hardcoded values, set the use `hardcoded_values` flag to True in the script. Alternatively, you can run the script with command-line arguments:

    ```bash
    python main.py EFO_ID Disease_Name
    ```
    Replace EFO_ID with the desired `EFO ID` and `Disease_Name` with the optional Disease Name if needed.

3. Explore the Jupyter Notebook:

    For a detailed research walkthrough and exploration, you can open this [research notbook](notebook/Fetching%20OpenTargets%20and%20STRING%20Database%20for%20Creating%20Gene-Protein%20Interaction%20Networks.ipynb) file. It provides an in-depth guide and visualizations to help you understand how the graph is composed and the data is processed.

## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
template_repo
├── CHANGELOG.md      <- Log of changes made
│
├── README.md
│
├── .gitignore        <- Specifies intentionally untracked files to ignore by git
│
├── configs           <- Dir to store config files. Conda env, requirements.txt, etc.
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
├── notebooks         <- Dir to store Jupyter, R Markdown notebooks, etc.
│
├── src               <- Dir to store source code for this project
```

## Team Structure and Contribution

Use this space to write your team names and their contribution.

## References

Provide references of repositories, articles, other work used by your teams.

[DO NOT REMOVE]

Template repo derived from: http://drivendata.github.io/cookiecutter-data-science

Template created by: @samuelbharti
