import os
import sys
import pytest
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.save_df_to_csv import save_df_to_csv


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'Col1': [1, 2, 3, 4],
        'Col2': ['A', 'B', 'C', 'D']
    })


@pytest.fixture
def empty_df():
    return pd.DataFrame()


# Test saving a DataFrame to CSV with index
def test_save_df_to_csv_success_with_index(valid_df, tmp_path):
    file_name = "test_df_with_index.csv"
    save_df_to_csv(valid_df, str(tmp_path), file_name)

    file_path = os.path.join(str(tmp_path), file_name)
    assert os.path.exists(file_path), f"File {file_path} should exist."

    loaded_df = pd.read_csv(file_path, index_col=0)
    pd.testing.assert_frame_equal(valid_df, loaded_df)


# Test saving a DataFrame to CSV without index
def test_save_df_to_csv_success_without_index(valid_df, tmp_path):
    file_name = "test_df_without_index.csv"
    save_df_to_csv(valid_df, str(tmp_path), file_name, index=False)

    file_path = os.path.join(str(tmp_path), file_name)
    assert os.path.exists(file_path), f"File {file_path} should exist."

    loaded_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(valid_df.reset_index(drop=True), loaded_df)


# Test invalid filename extension
def test_invalid_filename(valid_df, tmp_path):
    file_name = "test_file.txt"

    with pytest.raises(ValueError, match="Filename must end with '.csv'"):
        save_df_to_csv(valid_df, str(tmp_path), file_name)


# Test if input is not a DataFrame
def test_invalid_dataframe_type(tmp_path):
    invalid_df = "One day, a string can be a dataframe..."
    file_name = "test_file.csv"

    with pytest.raises(TypeError, match="The 'df' argument must be a pandas DataFrame."):
        save_df_to_csv(invalid_df, str(tmp_path), file_name)


# Test if DataFrame is empty
def test_empty_dataframe(empty_df, tmp_path):
    file_name = "test_empty.csv"

    with pytest.raises(ValueError, match="DataFrame must contain observations."):
        save_df_to_csv(empty_df, str(tmp_path), file_name)
