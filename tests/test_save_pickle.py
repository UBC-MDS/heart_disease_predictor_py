import os
import sys
import pytest
import pickle
from sklearn.dummy import DummyClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_pickle import save_pickle


# Temporary directory for testing
@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path


# Test saving a model as pickle
def test_save_pickle_file_existence(temp_directory):
    model = DummyClassifier(strategy="most_frequent")

    filename = "test_save_dummy.pickle"
    save_pickle(model, str(temp_directory), filename)

    file_path = os.path.join(str(temp_directory), filename)
    assert os.path.exists(file_path), f"File {file_path} should exist."


# Test if pickle content is saved correctly
def test_verify_pickle_content(temp_directory):
    model = DummyClassifier(strategy="most_frequent")
    filename = "test_save_dummy.pickle"
    save_pickle(model, str(temp_directory), filename)
    file_path = os.path.join(str(temp_directory), filename)

    with open(file_path, 'rb') as f:
        loaded_model = pickle.load(f)

    assert isinstance(loaded_model, DummyClassifier), (
        "Expected model to be a DummyClassifier."
    )
