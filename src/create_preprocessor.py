from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer

def create_thal_pipeline():
    """
    Create a pipeline for handling the 'thal' column.

    Returns
    ----------
    Pipeline
        Pipeline for imputing and one-hot encoding 'thal'.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(sparse_output=False)
    )

def create_ca_pipeline():
    """
    Create a pipeline for handling the 'ca' column.

    Returns
    ----------
    Pipeline
        Pipeline for imputing and scaling 'ca'.
    """
    return make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        StandardScaler()
    )

def create_preprocessor():
    """
    Create a column transformer for the heart disease dataset.

    Returns
    ----------
    ColumnTransformer
        Preprocessor for scaling, encoding, and imputing features.
    """
    ca_pipeline = create_ca_pipeline()
    thal_pipeline = create_thal_pipeline()

    return make_column_transformer(
        (ca_pipeline, ['ca']),
        (thal_pipeline, ['thal']),
        (OneHotEncoder(sparse_output=False), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope']),
        (StandardScaler(), ["age", "trestbps", "chol", "thalach", "oldpeak"]),
        remainder='passthrough',
        verbose_feature_names_out=True
    )