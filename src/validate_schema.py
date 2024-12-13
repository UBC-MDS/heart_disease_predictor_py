import pandera as pa
from pandera import DataFrameSchema

def validate_schema(data):
    """
    Validate the dataset against a predefined schema using `pandera`.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to validate.

    Returns
    ----------
    None
    """
    data["ca"] = data["ca"].astype('float64')
    schema = DataFrameSchema({
        "age": pa.Column(float, pa.Check.between(0, 100)),
        "sex": pa.Column(float, pa.Check.isin([0.0, 1.0])),
        "cp": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0, 4.0])),
        "trestbps": pa.Column(float, pa.Check.between(20, 220)),
        "chol": pa.Column(float, pa.Check.between(50, 800)),
        "fbs": pa.Column(float, pa.Check.isin([0.0, 1.0])),
        "restecg": pa.Column(float, pa.Check.isin([0.0, 1.0, 2.0])),
        "thalach": pa.Column(float, pa.Check.between(50, 240)),
        "exang":  pa.Column(float, pa.Check.isin([0.0, 1.0])),
        "oldpeak": pa.Column(float, pa.Check.between(0.0, 7.0)),
        "slope": pa.Column(float, pa.Check.isin([1.0, 2.0, 3.0])),
        "ca": pa.Column(float, pa.Check.between(0, 4), nullable=True),
        "thal": pa.Column(str, pa.Check.isin(["3.0", "6.0", "7.0"]), nullable=True),
        "num": pa.Column(int, pa.Check.between(0, 4))
    },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )

    try:
        schema.validate(data, lazy=True)
        print("Data validation passed successfully.")
    except pa.errors.SchemaErrors as e:
        print(e.failure_cases)
    print("------------------------")
