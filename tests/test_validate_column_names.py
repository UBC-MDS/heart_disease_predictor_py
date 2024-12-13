import pytest
import pandas as pd
import os
import sys
from io import StringIO
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from validate_column_names import validate_column_names 

def test_column_names_match():
    # Create a dataframe with matching column names
    data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c']
    })
    expected_columns = ['col1', 'col2']

    validate_column_names(data, expected_columns)

def test_column_names_mismatch():
    # Create a dataframe with column names that do not match
    data = pd.DataFrame({
        'col1': [1, 2, 3],
        'wrong_col': ['a', 'b', 'c']
    })
    expected_columns = ['col1', 'col2']

    validate_column_names(data, expected_columns)

def test_column_names_extra_column():
    # Create a dataframe with an extra column not expected
    data = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'extra_col': [4, 5, 6]
    })
    expected_columns = ['col1', 'col2']

    validate_column_names(data, expected_columns)

def test_column_names_missing_column():
    # Create a dataframe with a missing expected column
    data = pd.DataFrame({
        'col1': [1, 2, 3]
    })
    expected_columns = ['col1', 'col2']

    validate_column_names(data, expected_columns)