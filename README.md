# Heart Disease Predictor

## Authors
* Stephanie Wu
* Albert Halim
* Rongze Liu
* Ziyuan Zhao

----

## About

The **Heart Disease Predictor** project aims to build a reliable machine learning model that predicts the presence of heart disease based on patient health measurements. It involves data wrangling, exploratory data analysis (EDA), and classification techniques to find meaningful patterns and build an accurate model.

The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease) and contains 303 patient records with 13 attributes related to heart health. Our model’s goal is to predict heart disease presence to assist clinicians in assessing risk.

---

## Project Objectives
- **Data Wrangling**: Preprocess the raw dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Examine key relationships among features.
- **Model Development**: Train and evaluate a classification model to predict heart disease.
- **Evaluation**: Assess model performance using accuracy, confusion matrices, and other metrics.

Our current classifier achieves ~87% accuracy, though improvements could enhance clinical usefulness—especially reducing false negatives.

---

## Dataset Details

The dataset includes various features (e.g., age, sex, chest pain type, resting blood pressure, cholesterol, max heart rate) that have been used to study risk factors related to heart disease.

---

## Report
A summary of the findings and model development can be found in the [final report](https://ubc-mds.github.io/heart_disease_predictor_py/).

---

## Dependencies
- [Python 3.8+](https://www.python.org/downloads/)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)
- [Jupyter Notebook](https://jupyter.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scikit-Learn](https://scikit-learn.org/stable/)

A complete list of dependencies is in the `environment.yml` file.

---

## Installation and Setup

You can set up and run this project in two main ways:

1. **Local Environment (using Conda)**
2. **Using Docker (with or without Docker Compose)**

Choose the approach that best fits your environment.

### 1. Local Environment with Conda

#### Prerequisites
- Install [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.

#### Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/UBC-MDS/heart_disease_predictor_py.git
   cd heart_disease_predictor_py
   ```
   
2. **Create and Activate the Environment**  
   ```bash
   conda env create -f environment.yml
   conda activate heart_disease_predictor
   ```

3. **Run the Analysis (Make Targets)**  
   - To start fresh (remove previously generated files):
     ```bash
     make clean
     ```
   - To run the entire pipeline and produce the final outputs:
     ```bash
     make all
     ```
   
   This will download data, preprocess it, run EDA, train and evaluate models, and render the report.

### 2. Using Docker

If you prefer a containerized environment, use our pre-built Docker image that includes all dependencies.

#### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) installed on your system.

#### Steps

1. **Run the Docker Container**
Go to the root of this project in the terminal and then run:
   ```bash
   docker compose up
   ```
2. **Run the Analysis (Make Targets)**  
   - To start fresh (remove previously generated files):
     ```bash
     make clean
     ```
   - To run the entire pipeline and produce the final outputs:
     ```bash
     make all
     ```
   
---

## Running the Analysis Manually (Without Make)

If you prefer not to use `make`, you can manually run each step after setting up your environment (via Conda or Docker):

1. **Download the Data**
   ```bash
   python scripts/download_data.py \
       --url="https://archive.ics.uci.edu/static/public/45/heart_disease.zip" \
       --path="data/raw"
   ```

2. **Split and Preprocess the Data**
   ```bash
   python scripts/split_n_preprocess.py \
       --input-path=data/raw/processed.cleveland.data \
       --data-dir=data/processed \
       --preprocessor-dir=results/models \
       --seed=522
   ```

3. **Perform EDA**
   ```bash
   python scripts/script_eda.py \
       --input_data_path=data/processed/heart_disease_train.csv \
       --output_prefix=results/
   ```

4. **Fit the Predictive Models**
   ```bash
   python scripts/fit_heart_disease_predictor.py \
       --train-set=data/processed/heart_disease_train.csv \
       --preprocessor=results/models/heart_disease_preprocessor.pickle \
       --pipeline-to=results/models \
       --table-to=results/tables \
       --seed=522
   ```

5. **Evaluate the Models and Generate Figures**
   ```bash
   python scripts/evaluate_heart_disease_predictor.py \
       --test-set=data/processed/heart_disease_test.csv \
       --pipeline-svc-from=results/models/heart_disease_svc_pipeline.pickle \
       --pipeline-lr-from=results/models/heart_disease_lr_pipeline.pickle \
       --table-to=results/tables \
       --plot-to=results/figures \
       --seed=522
   ```

6. **Render the Report**
   ```bash
   quarto render report/heart_disease_predictor_report.qmd --to html
   quarto render report/heart_disease_predictor_report.qmd --to pdf
   ```

---

## Run the Tests

After ensuring that you are in the project root directory, you can run the tests in the terminal with the following command:
   ```bash
   pytest
   ```
This will execute all the test scripts located in the `tests/` directory within the Docker container.

---

## Updating Dependencies and Docker Image

1. **Add/Update Dependencies**  
   Edit `environment.yml` and then regenerate the conda lock file:
   ```bash
   conda-lock install --name heart_disease_predictor --file environment.yml
   ```

2. **Rebuild the Docker Image** (if using Docker)  
   ```bash
   docker build -t achalim/heart_disease_predictor_py:latest .
   docker push achalim/heart_disease_predictor_py:latest
   ```

---

## Clean Up

- To clean generated files (figures, models, tables):
  ```bash
  make clean
  ```

- To deactivate the conda environment:
  ```bash
  conda deactivate
  ```

- To stop and remove Docker containers, use Ctrl + C in the terminal and then run this:
  ```bash
  docker compose down
  ```

---


## License

All code in the Heart Disease Predictor project is licensed under the MIT License. The project report is licensed under the CC0 1.0 Universal License. If you use or re-mix any part of this project, please provide appropriate attribution.

## References

- Dua, D., Dheeru, D., & Graff, C. (2017). *UCI Machine Learning Repository*. University of California, Irvine. [https://archive.ics.uci.edu/ml](https://archive.ics.uci.edu/ml)
- Cleveland Clinic Foundation. (1988). Heart disease data set. In *Proceedings of Machine Learning and Medical Applications*.
- Attia, P. (2023, February 15). Peter on the four horsemen of chronic disease. PeterAttiaMD.com. [https://peterattiamd.com/peter-on-the-four-horsemen-of-chronic-disease/](https://peterattiamd.com/peter-on-the-four-horsemen-of-chronic-disease/)
- Bui, T. (2024, October 15). Cardiovascular disease is rising again after years of improvement. Stat News. [https://www.statnews.com/2024/10/15/cardiovascular-disease-rising-experts-on-causes/](https://www.statnews.com/2024/10/15/cardiovascular-disease-rising-experts-on-causes/)
- Centers for Disease Control and Prevention (CDC). (2022). Leading causes of death. National Center for Health Statistics. [https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm](https://www.cdc.gov/nchs/fastats/leading-causes-of-death.htm)
- Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1988). Heart Disease UCI dataset. UC Irvine Machine Learning Repository. [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Carlén, A., Gustafsson, M., Åström Aneq, M., & Nylander, E. (2019). Exercise-induced ST depression in an asymptomatic population without coronary artery disease. Scandinavian Cardiovascular Journal, 53(4), 206–212. https://doi.org/10.1080/14017431.2019.1626021
- Fuchs, F. D., & Whelton, P. K. (2020). High Blood Pressure and Cardiovascular Disease. Hypertension, 75(2), 285–292. https://doi.org/10.1161/HYPERTENSIONAHA.119.14240
- Regitz-Zagrosek, V., & Gebhard, C. (2023). Gender medicine: Effects of sex and gender on cardiovascular disease manifestation and outcomes. Nature Reviews Cardiology, 20(4), 236–247. https://doi.org/10.1038/s41569-022-00797-4
