import os
import sys
import pandas as pd
import numpy as np
import click
import requests
import zipfile
import warnings
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from load_data import load_data
from validate_column_names import validate_column_names
from check_missing_values import check_missing_values
from validate_schema import validate_schema
from process_target_variable import process_target_variable
from save_processed_data import save_processed_data
from create_preprocessor import create_preprocessor, create_ca_pipeline, create_thal_pipeline
from save_pickle import save_pickle
from scale_and_save_data import scale_and_save_data
from sklearn import set_config
from sklearn.model_selection import train_test_split

warnings.simplefilter(action='ignore', category=FutureWarning)

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
    save_pickle(heart_disease_preprocessor, preprocessor_dir, filename="heart_disease_preprocessor.pickle")

    # Fit and scale the datasets, then save them
    heart_disease_preprocessor.fit(heart_disease_train)
    scale_and_save_data(heart_disease_preprocessor, heart_disease_train, heart_disease_test, data_dir)

if __name__ == '__main__':
    main()
