# Heart Disease Predictor

## Authors
* [Stephanie Wu]
* [Albert Halim]
* [Rongze Liu]
* [Ziyuan Zhao]

----

## About

The **Heart Disease Predictor** project aims to build a reliable machine learning model that predicts the presence of heart disease based on a set of patient health measurements. This project employs data wrangling, exploratory data analysis (EDA), and classification techniques to explore the dataset and develop an accurate model.

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease). The dataset consists of 303 patient records, each including 13 attributes such as age, cholesterol levels, chest pain type, and maximum heart rate achieved. The target variable (`num`) indicates the presence or absence of heart disease. Our goal is to predict the target variable effectively, helping to assess patients' heart health in a clinical setting.

---

## Project Objectives
- **Data Wrangling**: Preprocess the raw dataset to prepare it for analysis.
- **Exploratory Data Analysis (EDA)**: Investigate relationships between patient features and heart disease presence.
- **Model Development**: Train and evaluate a classification model to predict heart disease.
- **Evaluation**: Assess the model's performance using metrics like accuracy, confusion matrices, and more.

Our final classifier achieved an overall accuracy of ~87%, which, while promising, indicates further improvements can be made for real-world applicability. False negatives (missed heart disease) remain a primary concern, as they could lead to underdiagnosis.

---

## Dataset Details

The heart disease dataset was originally collected by researchers from four different institutions and compiled by researchers at the Cleveland Clinic Foundation. The attributes in the dataset include:
- **Age**: Patient age in years.
- **Sex**: Gender of the patient.
- **Chest Pain Type (cp)**: Type of chest pain experienced (four categories).
- **Resting Blood Pressure (trestbps)**: Blood pressure at rest.
- **Cholesterol Level (chol)**: Serum cholesterol in mg/dl.
- **Max Heart Rate (thalach)**: Maximum heart rate achieved during exercise.

Additional features capture other physiological details, each potentially relevant to heart disease diagnosis.

---

## Report
The final report summarizing our findings and model development can be found [here](./src/heart_disease_predictor_report.ipynb).

---

## Dependencies
- [Python 3.8+](https://www.python.org/downloads/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [Jupyter Notebook](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

For a complete list of dependencies, refer to the `environment.yml` file.

---

## Setup Instructions

### Prerequisites
- Install [Conda](https://docs.conda.io/en/latest/miniconda.html) to handle dependencies.

### Setting Up the Environment

1. Clone this GitHub repository:
    ```bash
    git clone https://github.com/UBC-MDS/heart_disease_predictor_py.git
    cd heart_disease_predictor_py
    ```

2. Create and activate the environment:
    ```bash
    conda env create -f environment.yml
    conda activate heart_disease_env
    ```

3. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

## Usage

### Running the Analysis

1. Navigate to the root of this project on your computer using the command line.
2. Open the Jupyter notebook:
    ```bash
    src/heart_disease_predictor_report.ipynb
    ```
3. Execute the notebook cells to run the data wrangling, EDA, and modeling steps.

### Clean up
- To deactivate the environment:
    ```bash
    conda deactivate
    ```
### Adding a New Dependency

1. Add the new dependency to the `environment.yml` file in a separate branch.
2. Regenerate the `conda-lock` file:
    ```bash
    conda-lock -p linux-64 -p osx-64 --file environment.yml
    ```
3. Test the updated environment and push your changes.

## License

The Heart Disease Predictor project and its content are licensed under the MIT License. If you use or re-mix any part of this project, please provide appropriate attribution.

## References

- Dua, Dheeru, and Casey Graff. 2017. "UCI Machine Learning Repository." University of California, Irvine. [https://archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml).
- Cleveland Clinic Foundation. 1988. "Heart Disease Data Set." In *Proceedings of Machine Learning and Medical Applications*.
