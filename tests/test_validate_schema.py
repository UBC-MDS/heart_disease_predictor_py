import pytest
import pandas as pd
import pandera as pa
from pandera.errors import SchemaErrors
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from validate_schema import validate_schema 

# Sample valid data for the test
valid_data = pd.DataFrame({
    "age": [64.0, 48.0, 52.0],
    "sex": [0.0, 1.0, 1.0],
    "cp": [4.0, 2.0, 4.0],
    "trestbps": [130.0, 110.0, 125.0],
    "chol": [303.0, 229.0, 212.0],
    "fbs": [0.0, 0.0, 0.0],
    "restecg": [0.0, 0.0, 0.0],
    "thalach": [122.0, 168.0, 168.0],
    "exang": [0.0, 0.0, 0.0],
    "oldpeak": [2.0, 1.0, 1.0],
    "slope": [2.0, 2.0, 2.0],
    "ca": [2.0, 0.0, 2.0],
    "thal": ["3.0", "7.0", "7.0"],  # Ensure these are strings
    "num": [0, 1, 2]
})


# Sample invalid data for the test
invalid_age_data = pd.DataFrame({
    "age": [150, -10, 200],
    "sex": [0.0, 1.0, 0.0],
    "cp": [1.0, 2.0, 3.0],
    "trestbps": [120, 130, 110],
    "chol": [250, 230, 270],
    "fbs": [0.0, 1.0, 0.0],
    "restecg": [0.0, 1.0, 2.0],
    "thalach": [150, 160, 140],
    "exang": [0.0, 1.0, 0.0],
    "oldpeak": [1.2, 2.0, 1.5],
    "slope": [1.0, 2.0, 3.0],
    "ca": [0.0, 1.0, 2.0],
    "thal": ["3.0", "6.0", "7.0"],
    "num": [0, 1, 2]
})

invalid_sex_data = pd.DataFrame({
    "age": [50, 60, 45],
    "sex": [0.0, 1.0, 3.0],  # Invalid sex value (3.0 is not allowed)
    "cp": [1.0, 2.0, 3.0],
    "trestbps": [120, 130, 110],
    "chol": [250, 230, 270],
    "fbs": [0.0, 1.0, 0.0],
    "restecg": [0.0, 1.0, 2.0],
    "thalach": [150, 160, 140],
    "exang": [0.0, 1.0, 0.0],
    "oldpeak": [1.2, 2.0, 1.5],
    "slope": [1.0, 2.0, 3.0],
    "ca": [0.0, 1.0, 2.0],
    "thal": ["3.0", "6.0", "7.0"],
    "num": [0, 1, 2]
})

# Sample data with duplicate rows
duplicate_data = pd.DataFrame({
    "age": [50, 50],
    "sex": [0.0, 0.0],
    "cp": [1.0, 1.0],
    "trestbps": [120, 120],
    "chol": [250, 250],
    "fbs": [0.0, 0.0],
    "restecg": [0.0, 0.0],
    "thalach": [150, 150],
    "exang": [0.0, 0.0],
    "oldpeak": [1.2, 1.2],
    "slope": [1.0, 1.0],
    "ca": [0.0, 0.0],
    "thal": ["3.0", "3.0"],
    "num": [0, 0]
})


# Sample data with NaN values in all columns
nan_data = pd.DataFrame({
    "age": [None, None, None],
    "sex": [None, None, None],
    "cp": [None, None, None],
    "trestbps": [None, None, None],
    "chol": [None, None, None],
    "fbs": [None, None, None],
    "restecg": [None, None, None],
    "thalach": [None, None, None],
    "exang": [None, None, None],
    "oldpeak": [None, None, None],
    "slope": [None, None, None],
    "ca": [None, None, None],
    "thal": [None, None, None],
    "num": [None, None, None]
})


@pytest.mark.parametrize("data, expected_result", [
    (valid_data, True),
    (invalid_age_data, False),
    (invalid_sex_data, False),
    (duplicate_data, False),
    (nan_data, False)
])

def test_validate_schema(data, expected_result):
    """
    Test the validate_schema function with different datasets.
    """
    if expected_result:
        try:
            validate_schema(data)
        except pa.errors.SchemaErrors as e:
            pytest.fail(f"Validation failed unexpectedly: {e.failure_cases}")
    else:
        validate_schema(data)
