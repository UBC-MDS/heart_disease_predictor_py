import pandas as pd
import click
import os
import requests
import zipfile
import numpy as np
import pandera as pa
import pickle
from pandera import Column, Check, DataFrameSchema
import warnings
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(file_path, colnames):
    """
    Load and preprocess the heart disease dataset.
    
    Args:
        file_path (str): Path to the input dataset file (.data format).
        colnames (list of str): List of column names for the dataset.
    
    Returns:
        pd.DataFrame: Loaded and preprocessed dataset with missing values replaced by NaN.
    """
    if not file_path.endswith(".data"):
        print("Warning: The file extension is not .data. It should be a .data file.")
    else:
        print("File is in the expected format.")
    
    heart_disease = pd.read_csv(file_path, names=colnames, header=None)
    heart_disease.replace('?', np.nan, inplace=True)
    return heart_disease

def validate_column_names(heart_disease, colnames):
    """
    Validate if the dataset contains the expected column names.
    
    Args:
        heart_disease (pd.DataFrame): The dataset to validate.
        colnames (list of str): Expected column names.
    
    Returns:
        None: Prints warnings if column names do not match.
    """
    expected_columns = set(colnames)
    actual_columns = set(heart_disease.columns)
    if expected_columns != actual_columns:
        print(f"Warning: Column names do not match. Expected: {colnames}, Found: {heart_disease.columns.tolist()}")
    else:
        print("Column names are correct.")

def check_missing_values(heart_disease, acceptable_threshold=0.1):
    """
    Check for missing values in the dataset and validate against an acceptable threshold.
    
    Args:
        heart_disease (pd.DataFrame): The dataset to validate.
        acceptable_threshold (float): Threshold for acceptable proportion of missing values in any column.
    
    Returns:
        None: Prints warnings for missing values and columns exceeding the threshold.
    """
    missing_data = heart_disease.isna().sum().sum()
    if missing_data > 0:
        print(f"Warning: There are {missing_data} missing values in the dataset.")
    else:
        print("No missing values found in the dataset.")
    
    missing_proportions = heart_disease.isna().mean()
    for column, proportion in missing_proportions.items():
        if proportion > acceptable_threshold:
            print(f"Warning: Missing values in column '{column}' exceed the acceptable threshold ({proportion:.2%}).")
        else:
            print(f"Column '{column}' has acceptable missingness ({proportion:.2%}).")
    print("------------------------")

def validate_schema(heart_disease):
    """
    Validate the dataset against a predefined schema using `pandera`.
    
    Args:
        heart_disease (pd.DataFrame): The dataset to validate.
    
    Returns:
        None: Prints validation results, including schema errors if present.
    """
    heart_disease["ca"] = heart_disease["ca"].astype('float64')
    schema = DataFrameSchema({
        "age": pa.Column(float, pa.Check.between(0, 100)),  # Assuming ages are within a valid range
        "sex": pa.Column(float, pa.Check.isin([0.0, 1.0])),  # 0: female, 1: male
        "cp": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0, 4.0])),  # Chest pain types
        "trestbps": pa.Column(float, pa.Check.between(20, 220)),  # Resting blood pressure (reasonable range)
        "chol": pa.Column(float, pa.Check.between(50, 800)),  # Serum cholesterol (mg/dL)
        "fbs": pa.Column(float, pa.Check.isin([0.0, 1.0])),  # Fasting blood sugar > 120 mg/dL (0: no, 1: yes)
        "restecg": pa.Column(float, pa.Check.isin([0.0, 1.0, 2.0])),  # Resting ECG results
        "thalach": pa.Column(float, pa.Check.between(50, 240)),  # Maximum heart rate achieved
        "exang":  pa.Column(float, pa.Check.isin([0.0, 1.0])),  # Exercise-induced angina (0: no, 1: yes)
        "oldpeak": pa.Column(float, pa.Check.between(0.0, 7.0)),  # ST depression induced by exercise
        "slope": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0])),  # Slope of peak exercise ST segment
        "ca": pa.Column(float, pa.Check.between(0, 4), nullable=True),  # Number of major vessels colored by fluoroscopy
        "thal": pa.Column(str, pa.Check.isin(["3.0", "6.0", "7.0"]), nullable=True),  # Thalassemia types
        "num": pa.Column(int, pa.Check.between(0, 4))  # Diagnosis of heart disease (0: no disease to 4: severe disease)
    },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )
    
    try:
        schema.validate(heart_disease, lazy=True)
        print("Data validation passed successfully.")
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases)
    print("------------------------")

def process_target_variable(heart_disease):
    """
    Convert the target variable 'num' to binary format (1 if >1, else 0).
    
    Args:
        heart_disease (pd.DataFrame): The dataset to modify.
    
    Returns:
        None: Modifies the dataset in-place.
    """
    heart_disease['num'] = heart_disease['num'].apply(lambda x: 1 if x > 1 else x)

def save_processed_data(heart_disease_train, heart_disease_test, processed_dir):
    """
    Save processed training and testing datasets to the specified directory.
    
    Args:
        heart_disease_train (pd.DataFrame): Training dataset.
        heart_disease_test (pd.DataFrame): Testing dataset.
        processed_dir (str): Directory to save the datasets.
    
    Returns:
        None: Saves datasets as CSV files.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    heart_disease_train.to_csv(os.path.join(processed_dir, "heart_disease_train.csv"), index=False)
    heart_disease_test.to_csv(os.path.join(processed_dir, "heart_disease_test.csv"), index=False)

def create_ca_pipeline():
    """
    Create a pipeline for handling the 'ca' column.
    
    Returns:
        Pipeline: Pipeline for imputing and scaling 'ca'.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        StandardScaler()
    )

def create_thal_pipeline():
    """
    Create a pipeline for handling the 'thal' column.
    
    Returns:
        Pipeline: Pipeline for imputing and one-hot encoding 'thal'.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(sparse_output=False)
    )

def create_preprocessor():
    """
    Create a column transformer for the heart disease dataset.
    
    Returns:
        ColumnTransformer: Preprocessor for scaling, encoding, and imputing features.
    """
    ca_pipeline = create_ca_pipeline()
    thal_pipeline = create_thal_pipeline()
    
    return make_column_transformer(
        (ca_pipeline, ['ca']),  # Apply imputation and scaling to 'ca'
        (thal_pipeline, ['thal']),  # Apply imputation and encoding to 'thal'
        (OneHotEncoder(sparse_output=False), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']),
        (StandardScaler(), ["age", "trestbps", "chol", "thalach", "oldpeak"]),
        remainder='passthrough',
        verbose_feature_names_out=True
    )

def save_preprocessor(preprocessor, preprocessor_dir, filename="heart_disease_preprocessor.pickle"):
    """
    Save the preprocessor object to a pickle file.
    
    Args:
        preprocessor: Preprocessor to save.
        preprocessor_dir (str): Directory to save the preprocessor file.
        filename (str): Name of the pickle file. Defaults to 'heart_disease_preprocessor.pickle'.
    """
    if not os.path.exists(preprocessor_dir):
        os.makedirs(preprocessor_dir)
    pickle.dump(preprocessor, open(os.path.join(preprocessor_dir, filename), "wb"))

def scale_and_save_data(preprocessor, train_data, test_data, data_dir):
    """
    Scale the train and test datasets using the preprocessor and save them to CSV files.
    
    Args:
        preprocessor: Preprocessor to apply transformations.
        train_data (pd.DataFrame): Training dataset.
        test_data (pd.DataFrame): Testing dataset.
        data_dir (str): Directory to save the scaled datasets.
    """
    scaled_train = preprocessor.transform(train_data)
    scaled_test = preprocessor.transform(test_data)
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    scaled_train.to_csv(os.path.join(data_dir, "scaled_heart_disease_train.csv"), index=False)
    scaled_test.to_csv(os.path.join(data_dir, "scaled_heart_disease_test.csv"), index=False)


@click.command()
@click.option('--input-path', required=True, help="Path to input file")
@click.option('--data-dir', required=True, help="Path directory to processed data")
@click.option('--preprocessor-dir', required=True, help="Path directory to preprocessor")
@click.option('--seed', type=int, default=123, help="Random Seed")
def main(input_path, data_dir, preprocessor_dir, seed):
    """
    Main function to load, process, validate, and save the heart disease dataset.
    
    Args:
        input_path (str): Path to the input dataset file.
        data_dir (str): Directory to save the processed datasets.
        preprocessor_dir (str): Directory for the preprocessor result
    
    Returns:
        None: Executes the entire pipeline.
    """
    colnames = [
        "age",       
        "sex",       
        "cp",        
        "trestbps",  
        "chol",      
        "fbs",       
        "restecg",   
        "thalach",   
        "exang",     
        "oldpeak",   
        "slope",     
        "ca",        
        "thal",      
        "num"  
    ]
    
    heart_disease = load_data(input_path, colnames)
    validate_column_names(heart_disease, colnames)
    check_missing_values(heart_disease)
    validate_schema(heart_disease)
    process_target_variable(heart_disease)
    
    np.random.seed(seed)
    set_config(transform_output="pandas")
    heart_disease_train, heart_disease_test = train_test_split(
        heart_disease, train_size=0.70, stratify=heart_disease["num"]
    )
    
    save_processed_data(heart_disease_train, heart_disease_test, data_dir)

    # Create and save the preprocessor
    heart_disease_preprocessor = create_preprocessor()
    save_preprocessor(heart_disease_preprocessor, preprocessor_dir)

    # Fit and scale the datasets, then save them
    heart_disease_preprocessor.fit(heart_disease_train)
    scale_and_save_data(heart_disease_preprocessor, heart_disease_train, heart_disease_test, data_dir)

if __name__ == '__main__':
    main()
