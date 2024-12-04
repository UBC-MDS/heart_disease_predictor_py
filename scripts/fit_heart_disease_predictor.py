# fit_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-03

import os
import click
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from scipy.stats import loguniform
from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
warnings.filterwarnings('ignore', category=FutureWarning, module="deepchecks")


# Calculate the mean and std of cross validation
# This function is taken from UBC DSCI 571 Course
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores.iloc[i], std_scores.iloc[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


# Perform RandomizedSearchCV to find the best model
def randomized_search(X_train, y_train, model, param_dist, n_iter=100, cv=5, random_state=123):
    """
    Performs RandomizedSearchCV on the 
    specified model and returns the best model.
    
    Parameters:
    X_train : DataFrame
        Training features
    y_train : Series
        Training labels
    model : estimator
        The model to be tuned
    param_dist : dict
        Hyperparameter distribution for RandomizedSearchCV
    n_iter : int, optional, default=100
        Number of iterations for RandomizedSearchCV
    cv : int, optional, default=5
        Number of cross-validation folds
    random_state : int, optional, default=123
        Random seed for reproducibility

    Returns:
    best_model : estimator
        The best model after RandomizedSearchCV
    """
    # Perform RandomizedSearchCV
    random_search = RandomizedSearchCV(model, 
                                       param_distributions=param_dist,
                                       n_iter=n_iter, 
                                       cv=cv, 
                                       n_jobs=-1, 
                                       random_state=random_state,
                                       return_train_score=True)
    
    # Fit the model
    random_search.fit(X_train, y_train)

    # Return the best model found by RandomizedSearchCV
    return random_search.best_estimator_


# Save pandas DataFrame as image
def save_table_as_image(df: pd.DataFrame, width: int, height: int, output_dir: str, filename: str):
    """
    Saves a DataFrame as an image.
    
    Parameters:
    df : pandas.DataFrame
        The DataFrame to be saved as an image.
    width : int
        The width of the image.
    height : int
        The height of the image.
    plot_to : str
        The path where the image will be saved.
    filename : str
        The name of the image file to save.

    Returns:
    None
    """
    
    _, ax = plt.subplots(figsize=(width, height))

    # Adjust the outlook of the table
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values,
             rowLabels=df.index,
             colLabels=df.columns,
             loc='center')

    # Save the table as an image
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()