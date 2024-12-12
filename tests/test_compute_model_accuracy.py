import os
import sys
import pytest
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.compute_model_accuracy import compute_model_accuracy


@pytest.fixture
def pipeline():
    model = DummyClassifier(strategy="most_frequent")
    pipeline = Pipeline([('model', model)])
    return pipeline


@pytest.fixture
def test_set():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'num': [0, 1, 0]
    })


@pytest.fixture
def invalid_test_set():
    return 'This is not a test set but a string.'


@pytest.fixture
def empty_test_set():
    return pd.DataFrame(columns=['feature1', 'feature2', 'num'])


@pytest.fixture
def non_exist_target():
    return 'non_exist_target'


# Test computing model accuracy with a DummyClassifier
def test_compute_model_accuracy_success(pipeline, test_set):
    pipeline.fit(test_set.drop(columns=['num']), test_set['num'])
    accuracy = compute_model_accuracy(pipeline, test_set)
    assert isinstance(accuracy, float), "Accuracy should be a float."
    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1."


# Test if the input test_set is not a DataFrame
def test_compute_model_accuracy_invalid_test_set(pipeline, invalid_test_set):
    with pytest.raises(TypeError, match="The test_set must be a pandas DataFrame."):
        compute_model_accuracy(pipeline, invalid_test_set)


# Test if target_column doesn't exist in the test_set
def test_compute_model_accuracy_invalid_target(pipeline, test_set, non_exist_target):
    with pytest.raises(ValueError, match=f"The target '{non_exist_target}' doesn't exist in test_set."):
        compute_model_accuracy(
            pipeline,
            test_set,
            target_column=non_exist_target
        )


# Test if test_set is empty
def test_compute_model_accuracy_empty_test_set(pipeline, empty_test_set):
    with pytest.raises(ValueError, match="The test_set should not be empty."):
        compute_model_accuracy(pipeline, empty_test_set)


# Test if target_column is missing from the test_set
def test_compute_model_accuracy_missing_target_column(pipeline, test_set):
    test_set_no_target = test_set.drop(columns=['num'])
    with pytest.raises(ValueError, match="The target 'num' doesn't exist in test_set."):
        compute_model_accuracy(pipeline, test_set_no_target)
