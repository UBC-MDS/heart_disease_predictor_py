# Heart Disease Predictor

## Authors
* Stephanie Wu
* Albert Halim
* Rongze Liu
* Ziyuan Zhao

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
    conda activate heart_disease_predictor
    ```

3. Alternatively, create and activate the environment using `conda-lock`:
    ```bash
    conda-lock install --name heart_disease_predictor --file conda-lock.yml
    conda activate heart_disease_predictor
    ```

4. Start Jupyter Lab:
    ```bash
    jupyter lab
    ```

### Running the Analysis

1. Navigate to the root of this project on your computer using the command line.
2. Open the Jupyter notebook to start the analysis:
    ```bash
    jupyter lab src/heart_disease_predictor_report.ipynb
    ```
3. Execute the notebook cells to run the data wrangling, EDA, and modeling steps.
   - Make sure the kernel is set to the appropriate environment (`heart_disease_predictor`).
   - You can select "Restart Kernel and Run All Cells" from the "Kernel" menu to execute all steps in the analysis sequentially.

### Clean up
- To deactivate the environment:
    ```bash
    conda deactivate
    ```

### Adding a New Dependency

1. Add the new dependency to the `environment.yml` file in a separate branch.
2. Regenerate the `conda-lock` file:
    ```bash
    conda-lock install --name heart_disease_predictor --file environment.yml
    ```
3. Test the updated environment and push your changes.

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
