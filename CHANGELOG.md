# CHANGELOG

Here is a changelog format you could use. Feel free to choose a format that works for the repo and the pathway program.

## [Unreleased]
- Reorganized project structure for clarity and modularity
- Updated Dockerfile and .dockerignore for new structure and uv/Makefile usage
- Updated README to match new structure and Docker usage
- Added Makefile and tests to Docker context
- Removed legacy requirements.txt/start_app.sh references

```txt
[2024-08-09] Ahmed
* Added
    - Implemented `GraphComposer` class for fetching and processing gene-protein data.
    - Added methods for constructing and visualizing graphs from OpenTargets and STRING data.
    - Added a detailed notebook for teams to follow.

[2024-08-13] Ahmed
* Added
    - Retrieved the full Alzheimer’s disease dataset from the Open Targets platform using BigQuery.
    - Combined the Protein Details and Protein Info from the STRING database.
    - Conducted two experiments: one using a sample of the STRING database with the same length as the Alzheimer’s disease dataset, and another using the entire STRING database.
    - Applied various classifiers, including Logistic Regression and Random Forest.
    - Saved the model in the `models` directory.
```
---
