import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.generate_sum_statistics import generate_summary_statistics
from src.plot_histograms import plot_histograms
from src.plot_target_distribution import plot_target_distribution
from src.plot_cat_barplots import plot_categorical_stacked_barplots
from src.plot_numeric_boxplots import plot_numeric_boxplots
import click

@click.command()
@click.option('--input_data_path', type=str, required=True, help='Path to the input CSV data file')
@click.option('--output_prefix', type=str, required=True, help='Output directory prefix')
def perform_eda(input_data_path, output_prefix):
    """
    Main EDA driver function.
    """
    data = pd.read_csv(input_data_path)

    figures_dir = os.path.join(output_prefix, "figures")
    tables_dir = os.path.join(output_prefix, "tables")

    generate_summary_statistics(data, tables_dir)
    plot_histograms(data, figures_dir)
    plot_target_distribution(data, figures_dir)
    plot_categorical_stacked_barplots(data, figures_dir)
    plot_numeric_boxplots(data, figures_dir)

if __name__ == "__main__":
    perform_eda()
