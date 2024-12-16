import pytest
import os
import pandas as pd
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.plot_target_distribution import plot_target_distribution

@pytest.fixture
def valid_data():
    return pd.DataFrame({
        'num': [0, 1, 0, 1, 1]
    })

def test_plot_target_distribution_success(valid_data, tmp_path):
    figures_dir = tmp_path
    plot_target_distribution(valid_data, figures_dir)
    output_path = os.path.join(figures_dir, "eda_output_target_distribution.png")
    assert os.path.exists(output_path), "Plot file was not created."

def test_plot_target_distribution_empty_data(tmp_path):
    empty_data = pd.DataFrame()
    figures_dir = tmp_path
    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        plot_target_distribution(empty_data, figures_dir)
