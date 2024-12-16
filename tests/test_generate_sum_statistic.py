import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.generate_sum_statistics import generate_summary_statistics

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'Col1': [1, 2, 3, 4],
        'Col2': ['A', 'B', 'C', 'D']
    })

@pytest.fixture
def empty_df():
    return pd.DataFrame()

def test_generate_summary_statistics(valid_df, tmp_path):
    output_path = tmp_path / "summary_stats.csv"
    generate_summary_statistics(valid_df, output_path)
    assert output_path.exists(), "Summary statistics file should exist."

def test_generate_summary_statistics_empty_df(empty_df, tmp_path):
    output_path = tmp_path / "summary_stats_empty.csv"
    with pytest.raises(ValueError, match="DataFrame is empty"):
        generate_summary_statistics(empty_df, output_path)
