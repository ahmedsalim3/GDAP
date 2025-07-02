# GDAP - Gene-Disease Association Prediction App


## Requirements

To install and run the project, you will need:
- **Python 3.11+**: Ensure you have Python 3.11 or higher installed. You can download it from [python.org](https://www.python.org/).
- **uv**: Fast Python package installer and resolver. Install it from [uv.dev](https://uv.dev/).
- **Dependencies**: The project requires the Python packages listed in [pyproject.toml](./pyproject.toml) and locked in [uv.lock](./uv.lock).

## Quick Start

The easiest way to get started is using the provided Makefile:

```bash
# Install dependencies
make install

# Run the Streamlit app
make run-app

# Run the main script
make run-script
```

## How to install

1. **Clone the Repository** from terminal:
    ```bash
    git clone https://github.com/mentorchains/BI-ML_Disease-Prediction_2024.git
    cd BI-ML_Disease-Prediction_2024
    ```

2. **Install Dependencies using uv**:
    ```bash
    # Install production dependencies
    make install

    # Or install with development tools
    make install-dev
    ```

## How to Run the Script

1. Go to [Open Target Platform](https://platform.opentargets.org/) and obtain the disease `EFO ID`.

2. Update the configuration in [config.py](./src/config.py) for your experiment. If you choose BigQuery as a data source, ensure you set up your `GOOGLE_APPLICATION_CREDENTIALS` and follow the [steps](./docs/reports/google_cloud_setup.md) to obtain the necessary JSON key files.

3. From the project root, run the script:

    ```bash
    # Using Makefile (recommended)
    make run-script

    # Or using uv directly
    uv run python src/main.py
    ```

## How to Run the App

### Running Locally

1. Run the following command from the project root:

    ```bash
    # Using Makefile (recommended)
    make run-app

<<<<<<< Updated upstream
    # Or using uv directly
    uv run streamlit run app/app.py
=======
    # Or using streamlit directly
    streamlit run streamlit_app.py
>>>>>>> Stashed changes
    ```

## Running the App via Docker

To run the app using Docker, follow these steps:

1. Build and run the Docker container:

    ```bash
    # Using Makefile (recommended)
    make run-app-docker

    # Or manually
    docker build -t gdap .
    docker run -p 8501:8501 gdap
    ```

## Available Make Commands

```bash
# Installation
make install
make install-dev

# Development
make fix
make test
make lint
make format

# Running Applications
make run-script
make run-app

# Docker
make run-app-docker
make run-app-docker-dev

# Cleanup
make clean
make clean-docker
make clean-docker-all

# Help
make help
```

## Repo's directory structure

The directory structure below shows the nature of files/directories used in this repo.

```sh
GDAP/
<<<<<<< Updated upstream
=======
├── streamlit_app.py        <- Main Streamlit entry point for deployment
├── requirements.txt        <- Dependencies for Streamlit Cloud
>>>>>>> Stashed changes
├── app/                    <- Streamlit applications
├── src/                    <- Source code
│   ├── gdap/               <- Main package
│   ├── config.py           <- Configuration
│   └── main.py             <- Main script
├── data/                   <- Data files
├── docs/                   <- Documentation
├── notebook/               <- Jupyter notebooks
├── tests/                  <- Test files
├── pyproject.toml          <- Project config
└── Makefile                <- Build commands

```

## Team Structure and Contribution

@[ahmedsalim3](https://github.com/ahmedsalim3)

## References

Provide references of repositories, articles, other work used by your teams.

[DO NOT REMOVE]

Template repo derived from: http://drivendata.github.io/cookiecutter-data-science

Template created by: @samuelbharti
