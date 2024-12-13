import os
import numpy as np


def save_misclassified_examples(model, X_test, y_test, output_dir, filename):
    """
    Identify misclassified examples and save them to a CSV file.

    Parameters:
    ----------
    model: estimator
        The trained model.
    X_test: pandas.DataFrame
        The test set features.
    y_test: pandas.Series
        The true labels for the test set.
    output_dir: str
        The directory where the CSV file will be saved.
    filename: str
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
