# fit_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-04

import os
import click
import pickle
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV, cross_validate
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
def randomized_search_best(X_train, y_train, model, param_dist, n_iter=100, cv=5, random_state=123):
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


@click.command()
@click.option('--train-set', type=str, help="Path to train set")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path where the pipeline object will be saved")
@click.option('--table-to', type=str, help="Path where the table will be saved")
@click.option('--seed', type=int, help="Random seed for reproducibility", default=522)
def main(train_set, preprocessor, pipeline_to, table_to, seed):
    '''summary here
    '''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read training set and preprocessor
    heart_disease_train = pd.read_csv(train_set)
    heart_disease_preprocessor = pickle.load(open(preprocessor, "rb"))

    # Validate if there are anomalous correlations
    # between target and explanatory variables
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    heart_disease_ds = Dataset(heart_disease_train, label='num', cat_features=categorical_features)
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=heart_disease_ds)
    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-Target correlation exceeds the maximum acceptable threshold.")

    # Validate if there are anomalous correlations
    # between explanatory variables
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold = 0.9, n_pairs = 0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=heart_disease_ds)
    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-Feature correlation exceeds the maximum acceptable threshold.")
    
    # Modelling
    X_train = heart_disease_train.drop(columns=["num"])
    y_train = heart_disease_train["num"]
    results_dict = {}
    
    models = {
        "Dummy": DummyClassifier(random_state=123),
        "Decision tree": DecisionTreeClassifier(random_state=123),
        "SVC": SVC(random_state=123),
        "Logistic Regression": LogisticRegression(random_state=123, max_iter=1000)
    }

    for model in models.items():
        pipe = make_pipeline(heart_disease_preprocessor, model[1])
        results_dict[model[0]] = mean_std_cross_val_scores(
            pipe, X_train, y_train, cv=5, return_train_score=True
        )
    
    # Cross-validation results of baseline models
    cv_results_df = pd.DataFrame(results_dict).T

    # Save the validation scores as a csv
    cv_results_df.to_csv(os.path.join(table_to, "baseline_cv_results.csv"))

    # Models and their parameter grids
    models_params = {
        "Decision Tree": {
            "pipe": Pipeline(steps=[
                ("preprocessor", heart_disease_preprocessor),
                ("dt", DecisionTreeClassifier(random_state=123))
            ]),
            "param_grid": {
                "dt__max_depth": [i for i in range(1, 101)],
                "dt__class_weight": [None, "balanced"]
            }
        },
        "SVC": {
            "pipe": Pipeline(steps=[
                ("preprocessor", heart_disease_preprocessor),
                ("svc", SVC(random_state=123))
            ]),
            "param_grid": {
                "svc__gamma": loguniform(1e-4, 1e3),
                "svc__C": loguniform(1e-4, 1e3),
                "svc__class_weight": [None, "balanced"]
            }
        },
        "Logistic Regression": {
            "pipe": Pipeline(steps=[
                ("preprocessor", heart_disease_preprocessor),
                ("lr", LogisticRegression(random_state=123, max_iter=1000))
            ]),
            "param_grid": {
                "lr__C": loguniform(1e-4, 1e3),
                "lr__class_weight": [None, "balanced"]
            }
        }
    }
    
    # Tune models and store the best-model pipelines
    best_model_pipes = {
        model_name: randomized_search_best(X_train, y_train, model["pipe"], model["param_grid"])
        for model_name, model in models_params.items()
    }
    
    # Evaludate the tuned models using cross-validation
    results_dict = {
        model_name: mean_std_cross_val_scores(pipe, X_train, y_train, cv=5, return_train_score=True)
        for model_name, pipe in best_model_pipes.items()
    }
    
    # Cross-validation results for each best-model pipeline
    best_model_cv_results_df = pd.DataFrame(results_dict).T
    
    # Save the validation scores as a csv
    best_model_cv_results_df.to_csv(os.path.join(table_to, "best_model_cv_results.csv"))
    
    # Export both fitted SVC and LR model with pickle
    best_svc_model = best_model_pipes["SVC"]
    best_lr_model = best_model_pipes["Logistic Regression"]
    
    with open(os.path.join(pipeline_to, "heart_disease_svc_pipeline.pickle"), 'wb') as f:
        pickle.dump(best_svc_model, f)
    with open(os.path.join(pipeline_to, "heart_disease_lr_pipeline.pickle"), 'wb') as f:
        pickle.dump(best_lr_model, f)


if __name__ == "__main__":
    main()
