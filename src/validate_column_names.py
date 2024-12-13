import pandas as pd

def validate_column_names(data, colnames):
    """
    Validate if the dataset contains the expected column names.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to validate.
    colnames : list of str
        Expected column names.

    Returns
    ----------
    None
    """
    expected_columns = set(colnames)
    actual_columns = set(data.columns)
    if expected_columns != actual_columns:
        print(f"Warning: Column names do not match. Expected: {colnames}, Found: {data.columns.tolist()}")
    else:
        print("Column names are correct.")
