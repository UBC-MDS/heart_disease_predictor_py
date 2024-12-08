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

### Using the Docker Container

To simplify the setup process, we have created a Docker container that includes all necessary dependencies for the Heart Disease Predictor project. Follow the steps below to use the container:

1. **Pull the Docker Image**
   - Make sure Docker is installed on your machine. You can pull the latest version of the Docker image from DockerHub by running:
     ```bash
     docker pull achalim/heart_disease_predictor_py:latest
     ```

2. **Run the Docker Container**
   - To start a container instance using the pulled image, run:
     ```bash
     docker run -p 8888:8888 -v $(pwd):/home/jovyan/work achalim/heart_disease_predictor_py:latest
     ```
     - This will start a Jupyter Notebook server that you can access in your browser at `http://localhost:8888`.
     - The `-v $(pwd):/home/jovyan/work` option mounts your current directory into the container so that you can access your project files.

3. **Using Jupyter Lab**
   - Once the container is running, Jupyter Lab should open in your browser. You can run the analysis by navigating to `src/heart_disease_predictor_report.ipynb` and executing the cells as you would on your local setup.


### Running the Analysis

1. Navigate to the root of this project on your computer using the command line.
2. To run the analysis, run the following commands:
```
python scripts/download_data.py --url="https://archive.ics.uci.edu/static/public/45/heart+disease.zip" --path="data/raw"

python scripts/split_n_preprocess.py \
--input-path=data/raw/processed.cleveland.data \
--data-dir=data/processed \
--preprocessor-dir=results/models \
--seed=522

python scripts/eda.py --input_data_path data/processed/heart_disease_train.csv --output_prefix results/

python scripts/fit_heart_disease_predictor.py \
    --train-set=data/processed/heart_disease_train.csv \
    --preprocessor=results/models/heart_disease_preprocessor.pickle \
    --pipeline-to=results/models \
    --table-to=results/tables \
    --seed=522

python scripts/evaluate_heart_disease_predictor.py \
    --test-set=data/processed/heart_disease_test.csv \
    --pipeline-svc-from=results/models/heart_disease_svc_pipeline.pickle \
    --pipeline-lr-from=results/models/heart_disease_lr_pipeline.pickle \
    --table-to=results/tables \
    --seed=522
```
3. To render the Quarto markdown file to html and pdf, use the following commands:
```
quarto render src/heart_disease_predictor_report.qmd --to html
quarto render src/heart_disease_predictor_report.qmd --to pdf
```
 
 
### Updating the Docker Container

If there are changes in the codebase or dependencies, follow the steps below to update the container:

1. **Update the Dependencies**
   - If any changes are made to the `environment.yml` file, you must regenerate the `conda-lock` file to pin the versions of the updated dependencies:
     ```bash
     conda-lock install --name heart_disease_predictor --file environment.yml
     ```

2. **Rebuild the Docker Image**
   - Make sure the updated `environment.yml` and `Dockerfile` reflect the latest changes, then rebuild the Docker image using the command:
     ```bash
     docker build -t achalim/heart_disease_predictor_py:latest .
     ```

3. **Push the Updated Image**
   - To make the updated image available to others, push it to DockerHub:
     ```bash
     docker push achalim/heart_disease_predictor_py:latest
     ```

### Using Docker Compose 

To simplify running multiple containers or configuring ports/volumes, Docker Compose can be used. Here is how you can use Docker Compose:

1. **Docker Compose File**
   - Create a `docker-compose.yml` file in the root of your repository that defines the services required:
     ```yaml
     version: '3'
     services:
       heart_disease_predictor:
         image: achalim/heart_disease_predictor_py:latest
         ports:
           - "8888:8888"
         volumes:
           - .:/home/jovyan/work
     ```

2. **Running with Docker Compose**
   - Use the following command to launch the container with Docker Compose:
     ```bash
     docker-compose up
     ```
   - This will start the container, mapping the necessary ports and volumes as specified in the `docker-compose.yml` file.



### Clean up
- To exit the container and delete all the resources, type Ctrl + C in the terminal, and then type
  ```bash
  docker compose rm
  ```
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
