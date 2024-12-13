import os
import sys
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from sklearn.pipeline import make_pipeline
from src.mean_std_cross_val_scores import mean_std_cross_val_scores


def evaluate_models(models, preprocessor, X_train, y_train, cv=5):
    """
    Evaluate models with cross-validation and return results as a DataFrame.

    Parameters
    ----------
    models : dict
        A dictionary, keys are model names, values are model objects.
    preprocessor : object
        The preprocessor to be applied to the features.
    X_train : pandas.DataFrame
        The features.
    y_train : pandas.Series
        The target.
    cv : int, optional
        The number of cv folds. Default 5.

    Returns
    -------
    pandas.DataFrame
    """
    results_dict = {}

    # Iterate through models to compute cross-validation scores
    for model_name, model in models.items():
        pipe = make_pipeline(preprocessor, model)
        results_dict[model_name] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=cv, return_train_score=True
        )
    # Convert into DataFrame
    cv_results_df = pd.DataFrame(results_dict).T

    return cv_results_df
