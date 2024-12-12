# evaluate_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-12


import os
import sys
import click
import pandas as pd
import numpy as np
from sklearn import set_config
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_pickle import load_pickle
from src.save_df_to_csv import save_df_to_csv
from src.compute_model_accuracy import compute_model_accuracy
from src.save_feature_coefficients_plot import save_feature_coefficients_plot
from src.save_misclassified_examples import save_misclassified_examples


@click.command()
@click.option('--test-set', type=str, help="Path to test set")
@click.option('--pipeline-svc-from', type=str, help="Path to the SVC pipeline object")
@click.option('--pipeline-lr-from', type=str, help="Path to the LR pipeline object")
@click.option('--table-to', type=str, help="Path where the tables will be saved")
@click.option('--plot-to', type=str, help="Path where the figures will be saved")
@click.option('--seed', type=int, help="Random seed for reproducibility", default=522)
def main(test_set, pipeline_svc_from, pipeline_lr_from, table_to, plot_to, seed):
    '''Evaluates the performance of the heart disease
    model on the test set and saves the outputs.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read testing set and the pipeline objects
    heart_disease_test = pd.read_csv(test_set)
    heart_disease_svc_fit = load_pickle(pipeline_svc_from)
    heart_disease_lr_fit = load_pickle(pipeline_lr_from)

    # Compute accuracy
    test_scores = pd.DataFrame({
        "Model": ["SVC fitted", "Logistic Regression fitted"],
        "Accuracy": [
            compute_model_accuracy(heart_disease_svc_fit, heart_disease_test),
            compute_model_accuracy(heart_disease_lr_fit, heart_disease_test)
        ]
    })
    save_df_to_csv(test_scores, table_to, "test_score.csv", False)

    # Create and sort DataFrame by feature coefficients
    coef_df = pd.DataFrame({
        'Feature': heart_disease_lr_fit.named_steps['preprocessor'].get_feature_names_out(),
        'Coefficient': heart_disease_lr_fit.named_steps['lr'].coef_[0]
    }).sort_values(by='Coefficient', ascending=False).reset_index(drop=True)

    save_df_to_csv(coef_df, table_to, "coefficient_df.csv")

    # Plot the coefficients bar chart
    save_feature_coefficients_plot(coef_df, plot_to, "log_reg_feature_coefficients.png")

    # Save the misclassified entries as csv
    save_misclassified_examples(
        heart_disease_lr_fit,
        heart_disease_test.drop(columns=["num"]),
        heart_disease_test["num"],
        table_to,
        "misclassified_examples.csv"
    )


if __name__ == "__main__":
    main()
