from sklearn.model_selection import RandomizedSearchCV


def randomized_search_best(X_train, y_train, model, param_dist, n_iter=100, cv=5, random_state=123):
    """
    Performs RandomizedSearchCV on the 
    specified model and returns the best model.

    Parameters:
    ----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
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
    ----------
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
