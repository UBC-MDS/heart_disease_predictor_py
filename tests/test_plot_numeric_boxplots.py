import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plot_numeric_boxplots import plot_numeric_boxplots

@pytest.fixture
def valid_df():
    return pd.DataFrame({
        'num': [0, 1, 0, 1],
        'Col1': [10, 20, 15, 25],
        'Col2': [100, 200, 150, 250]
    })

@pytest.fixture
def empty_df():
    return pd.DataFrame()

def test_plot_numeric_boxplots(valid_df, tmp_path):
    figures_dir = tmp_path
    plot_numeric_boxplots(valid_df, figures_dir)
    output_path = os.path.join(figures_dir, "eda_output_numeric_boxplots.png")
    assert os.path.exists(output_path), "Numeric boxplots file should exist."

def test_plot_numeric_boxplots_empty_df(empty_df, tmp_path):
    figures_dir = tmp_path
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        plot_numeric_boxplots(empty_df, figures_dir)