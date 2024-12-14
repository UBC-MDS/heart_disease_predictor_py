import os
import sys
import pandas as pd
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def load_data(file_path, colnames):
    """
    Load and preprocess a dataset.

    Parameters
    ----------
    file_path : str
        Path to the input dataset file.
    colnames : list of str
        List of column names for the dataset.

    Returns
    ----------
    pd.DataFrame
        Loaded and preprocessed dataset with missing values replaced by NaN.
    """
    if not file_path.endswith(".data"):
        print("Warning: The file extension is not .data. It should be a .data file.")
    else:
        print("File is in the expected format.")

    data = pd.read_csv(file_path, names=colnames, header=None)
    data.replace('?', np.nan, inplace=True)
    return data