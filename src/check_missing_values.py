import os
import pandas as pd

def check_missing_values(data, acceptable_threshold=0.1):
    """
    Check for missing values in the dataset and validate against an acceptable threshold.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to validate.
    acceptable_threshold : float
        Threshold for acceptable proportion of missing values in any column.

    Returns
    ----------
    None
    """
    missing_data = data.isna().sum().sum()
    if missing_data > 0:
        print(f"Warning: There are {missing_data} missing values in the dataset.")
    else:
        print("No missing values found in the dataset.")

    missing_proportions = data.isna().mean()
    for column, proportion in missing_proportions.items():
        if proportion > acceptable_threshold:
            print(f"Warning: Missing values in column '{column}' exceed the acceptable threshold ({proportion:.2%}).")
        else:
            print(f"Column '{column}' has acceptable missingness ({proportion:.2%}).")
    print("------------------------")
