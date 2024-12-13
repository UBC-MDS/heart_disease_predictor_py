import os
import sys
import pytest
import pickle
from sklearn.dummy import DummyClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.load_pickle import load_pickle


@pytest.fixture
def temp_directory(tmp_path):
    return tmp_path


@pytest.fixture
def valid_pickle(temp_directory):
    model = DummyClassifier(strategy="most_frequent")
    pickle_path = os.path.join(temp_directory, "test_dummy_classifier.pickle")
    with open(pickle_path, 'wb') as f:
        pickle.dump(model, f)
    return pickle_path


@pytest.fixture
def invalid_txt(temp_directory):
    txt_file = os.path.join(temp_directory, "test_dummy_classifier.txt")
    with open(txt_file, 'w') as f:
        f.write("Gotcha, this is not a pickle! :D")
    return txt_file


# Test loading a valid pickle file
def test_load_pickle_success(valid_pickle):
    model = load_pickle(valid_pickle)
    assert isinstance(model, DummyClassifier), (
        "Expected model to be a DummyClassifier."
    )


# Test loading an invalid txt file
def test_load_invalid_file(invalid_txt):
    with pytest.raises(ValueError, match="Filename must end with '.pickle'."):
        load_pickle(invalid_txt)


# Test loading a pickle file that does not exist
def test_load_non_exist_file(temp_directory):
    non_exist_file = os.path.join(temp_directory, "non_exist.pickle")
    with pytest.raises(FileNotFoundError, match="The directory or file does not exist."):
        load_pickle(non_exist_file)


# Test loading a pickle file from a directory that does not exist
def test_load_non_exist_dir():
    non_exist_dir_file = "non_exist_dir/test_dummy_classifier.pickle"
    with pytest.raises(FileNotFoundError, match="The directory or file does not exist."):
        load_pickle(non_exist_dir_file)
