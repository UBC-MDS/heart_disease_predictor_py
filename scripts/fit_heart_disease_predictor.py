# fit_heart_disease_predictor.py
# author: Archer Liu
# date: 2024-12-12

import os
import sys
import click
import numpy as np
import pandas as pd
import warnings
from sklearn import set_config
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from scipy.stats import loguniform
warnings.filterwarnings('ignore', category=FutureWarning, module="deepchecks")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_pickle import load_pickle
from src.save_pickle import save_pickle
from src.save_df_to_csv import save_df_to_csv
from src.validate_corr import val_feat_label_corr, val_feat_feat_corr
from src.mean_std_cross_val_scores import mean_std_cross_val_scores
from src.evaluate_models import evaluate_models
from src.randomized_search_best import randomized_search_best


@click.command()
@click.option('--train-set', type=str, help="Path to train set")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path where the pipeline object will be saved")
@click.option('--table-to', type=str, help="Path where the table will be saved")
@click.option('--seed', type=int, help="Random seed for reproducibility", default=522)
def main(train_set, preprocessor, pipeline_to, table_to, seed):
    '''Fits a heart disease predictor using the train set
    and stores the resulting pipeline object.'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read training set and preprocessor
    heart_disease_train = pd.read_csv(train_set)
    heart_disease_preprocessor = load_pickle(preprocessor)

    # Validate if there are anomalous correlations
    # between target and explanatory variables
    val_feat_label_corr(heart_disease_train, target='num')

    # Validate if there are anomalous correlations
    # between explanatory variables
    val_feat_feat_corr(heart_disease_train, target='num')

    # Modelling
    X_train = heart_disease_train.drop(columns=["num"])
    y_train = heart_disease_train["num"]

    models = {
        "Dummy": DummyClassifier(random_state=123),
        "Decision tree": DecisionTreeClassifier(random_state=123),
        "SVC": SVC(random_state=123),
        "Logistic Regression": LogisticRegression(random_state=123, max_iter=1000)
    }

    # Cross-validation results of baseline models
    cv_results_df = evaluate_models(
        models,
        heart_disease_preprocessor,
        X_train,
        y_train,
        cv=5
    )

    # Save the validation scores as a csv
    save_df_to_csv(cv_results_df, table_to, "baseline_cv_results.csv")

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
    save_df_to_csv(best_model_cv_results_df, table_to, "best_model_cv_results.csv")

    # Export both fitted SVC and LR model with pickle
    save_pickle(
        best_model_pipes["SVC"],
        pipeline_to,
        "heart_disease_svc_pipeline.pickle"
    )
    save_pickle(
        best_model_pipes["Logistic Regression"],
        pipeline_to,
        "heart_disease_lr_pipeline.pickle"
    )


if __name__ == "__main__":
    main()
