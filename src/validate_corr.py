from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation


# Validate feature-label correlation
def val_feat_label_corr(data, target, threshold=0.9):
    """
    Validate feature-label correlation using Deepchecks.

    Parameters
    ----------
    data : pandas.DataFrame
        The train set.
    target : str
        The target column name.
    threshold : float
        The maximum correlation threshold.

    Raises
    ------
    ValueError if feature-label correlation exceeds the threshold.
    """
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    ds = Dataset(data, label=target, cat_features=categorical_features)
    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(threshold)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=ds)
    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-Target correlation exceeds the maximum acceptable threshold.")


# Validate feature-feature correlation
def val_feat_feat_corr(data, target, threshold=0.9, n_pairs=0):
    """
    Validate feature-feature correlation using Deepchecks.

    Parameters
    ----------
    data : pandas.DataFrame
        The train set.
    target : str
        The target column name.
    threshold : float
        The maximum correlation threshold.
    n_pairs : int
        The number of feature pairs that can exceed the threshold.

    Raises
    ------
    ValueError if feature-feature correlation exceeds the threshold.
    """
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    ds = Dataset(data, label=target, cat_features=categorical_features)
    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(
        threshold=threshold, n_pairs=n_pairs
    )
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=ds)
    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-Feature correlation exceeds the maximum acceptable threshold.")
