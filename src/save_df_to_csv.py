import os
import pandas as pd


def save_df_to_csv(df, dir_path, file_name, index=True):
    """
    Save a DataFrame to a CSV file in a specified directory.

    Parameters:
    ----------
    df: pandas.DataFrame
        The DataFrame to save.
    dir_path: str
        The directory where the CSV file will be saved.
    file_name: str
        The name of the CSV file.
    index: bool
        Whether to keep index or not.

    Returns
    ----------
    None

    Raises
    ----------
    ValueError
        If the filename does not end with '.csv', or the DataFrame is empty.
    TypeError
        If the input is not a pandas DataFrame.
    """
    if not file_name.endswith(".csv"):
        raise ValueError("Filename must end with '.csv'")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("The 'df' argument must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("DataFrame must contain observations.")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(os.path.join(dir_path, file_name), index=index)
