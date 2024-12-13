
import os
import pandas as pd

def save_processed_data(heart_disease_train, heart_disease_test, processed_dir):
    """
    Save processed training and testing datasets to the specified directory.

    Parameters
    ----------
    heart_disease_train : pd.DataFrame
        Training dataset.
    heart_disease_test : pd.DataFrame
        Testing dataset.
    processed_dir : str
        Directory to save the datasets.

    Returns
    ----------
    None
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    heart_disease_train.to_csv(os.path.join(processed_dir, "heart_disease_train.csv"), index=False)
    heart_disease_test.to_csv(os.path.join(processed_dir, "heart_disease_test.csv"), index=False)