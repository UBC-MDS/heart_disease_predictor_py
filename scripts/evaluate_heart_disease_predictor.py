# evaluate_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-07


import os
import click
import pickle
import pandas as pd
import numpy as np
from sklearn import set_config
import matplotlib.pyplot as plt


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
    features = test_set.drop(columns=[target_column])
    target = test_set[target_column]
    return pipeline.score(features, target)


# Save data frame as csv
def save_df_to_csv(df, dir_path, file_name, index=True):
    """
    Save a DataFrame to a CSV file in a specified directory.

    Parameters:
    ----------
    - df: pandas.DataFrame
        The DataFrame to save.
    - dir_path: str
        The directory where the CSV file will be saved.
    - file_name: str
        The name of the CSV file.
    - index: bool
        Whether to keep index or not.

    Returns
    ----------
    None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    df.to_csv(os.path.join(dir_path, file_name), index=index)


# Save feature coefficient plot as figures
def save_feature_coefficients_plot(coef_df, dir_path, filename):
    """
    Save a bar plot of feature coefficients for logistic regression.

    Parameters:
    ----------
    - coef_df: pandas.DataFrame
        The DataFrame containing 'Feature' and 'Coefficient' columns.
    - dir_path: str
        The directory where the plot will be saved.
    - filename: str
        The name of the saved image file.

    Returns
    ----------
    None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.figure(figsize=(12, 8))
    plt.barh(
        coef_df['Feature'],
        coef_df['Coefficient'],
        color=coef_df['Coefficient'].apply(
            lambda x: 'orange' if x > 0 else 'steelblue'
        )
    )
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Coefficients for Logistic Regression')

    plt.tight_layout()
    plt.savefig(f"{dir_path}/{filename}", dpi=300)
    plt.close()


def save_misclassified_examples(model, X_test, y_test, output_dir, filename):
    """
    Identify misclassified examples and save them to a CSV file.

    Parameters:
    ----------
    - model: estimator
        The trained model.
    - X_test: pandas.DataFrame
        The test set features.
    - y_test: pandas.Series
        The true labels for the test set.
    - output_dir: str
        The directory where the CSV file will be saved.
    - filename: str
        The name of the output CSV file.

    Returns
    ----------
    None
    """
    # Predict the labels for the test set
    predictions = model.predict(X_test)

    # Identify misclassified indices
    misclassified_indices = np.where(y_test != predictions)[0]

    # Create a DataFrame for misclassified examples
    misclassified_df = X_test.iloc[misclassified_indices].copy()
    misclassified_df['True Label'] = y_test.iloc[misclassified_indices].values
    misclassified_df['Predicted Label'] = predictions[misclassified_indices]

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the misclassified examples to a CSV file
    misclassified_df.to_csv(os.path.join(output_dir, filename), index=False)


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
