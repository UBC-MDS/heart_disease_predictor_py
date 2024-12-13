import os
import pandas as pd
from sklearn.compose import ColumnTransformer

def scale_and_save_data(preprocessor, train_data, test_data, data_dir):
    """
    Scale the train and test datasets using the preprocessor and save them to CSV files.

    Parameters
    ----------
    preprocessor : object
        Preprocessor to apply transformations.
    train_data : pd.DataFrame
        Training dataset.
    test_data : pd.DataFrame
        Testing dataset.
    data_dir : str
        Directory to save the scaled datasets.

    Returns
    ----------
    None
    """
    scaled_train = preprocessor.transform(train_data)
    scaled_test = preprocessor.transform(test_data)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    scaled_train.to_csv(os.path.join(data_dir, "scaled_heart_disease_train.csv"), index=False)
    scaled_test.to_csv(os.path.join(data_dir, "scaled_heart_disease_test.csv"), index=False)