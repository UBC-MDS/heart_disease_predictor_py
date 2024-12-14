def process_target_variable(data):
    """
    Convert the target variable 'num' to binary format (1 if >1, else 0).

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to modify.

    Returns
    ----------
    None
    """
    data['num'] = data['num'].apply(lambda x: 1 if x > 1 else x)