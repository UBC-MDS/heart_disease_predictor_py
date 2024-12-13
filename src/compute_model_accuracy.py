# Compute model accuracy
import pandas as pd


def compute_model_accuracy(pipeline, test_set, target_column="num"):
    """
    Compute the accuracy of a model on the test set.

    Parameters:
    ----------
    pipeline:
        Pipeline with the trained model
    test_set:
        Test set as DataFrame
    target_column:
        Name of the target in the test set

    Returns:
    ----------
        Accuracy score as a float.
    """
    if not isinstance(test_set, pd.DataFrame):
        raise TypeError("The test_set must be a pandas DataFrame.")

    if target_column not in test_set.columns:
        raise ValueError(f"The target '{target_column}' doesn't exist in test_set.")

    if test_set.empty:
        raise ValueError("The test_set should not be empty.")

    features = test_set.drop(columns=[target_column])
    target = test_set[target_column]
    return pipeline.score(features, target)
