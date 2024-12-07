# evaluate_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-05


import os
import click
import pickle
import pandas as pd
import numpy as np
from sklearn import set_config


# Load pipeline from path
def load_pipeline(path):
    """
    Load a pipeline object from a given file path.

    Parameters
    ----------
    path : str
        The file path to the pipeline object.

    Returns
    ----------
    object
        The trained pipeline object.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


# Compute model accuracy
def compute_model_accuracy(pipeline, test_set, target_column="num"):
    """
    Compute the accuracy of a model on the test set.

    Parameters:
        pipeline: Pipeline with the trained model
        test_set: Test set as DataFrame
        target_column: Name of the target in the test set

    Returns:
        Accuracy score as a float.
    """
    features = test_set.drop(columns=[target_column])
    target = test_set[target_column]
    return pipeline.score(features, target)


@click.command()
@click.option('--scaled-test-set', type=str, help="Path to scaled test set")
@click.option('--pipeline-svc-from', type=str, help="Path to the SVC pipeline object")
@click.option('--pipeline-lr-from', type=str, help="Path to the LR pipeline object")
@click.option('--results-to', type=str, help="Path where the results will be saved")
@click.option('--seed', type=int, help="Random seed for reproducibility", default=522)
def main(scaled_test_set, pipeline_svc_from, pipeline_lr_from, results_to, seed):
    '''Evaluates the performance of the heart disease
    model on the test set and saves the outputs.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read testing set and the pipeline objects
    heart_disease_test = pd.read_csv(scaled_test_set)
    heart_disease_svc_fit = load_pipeline(pipeline_svc_from)
    heart_disease_lr_fit = load_pipeline(pipeline_lr_from)

    # Compute accuracy
    test_scores = pd.DataFrame({
        "Model": ["SVC fitted", "Logistic Regression fitted"],
        "Accuracy": [
            compute_model_accuracy(heart_disease_svc_fit, heart_disease_test),
            compute_model_accuracy(heart_disease_lr_fit, heart_disease_test)
        ]
    })
    test_scores.to_csv(os.path.join(results_to, "test_scores.csv"), index=False)


if __name__ == "__main__":
    main()
