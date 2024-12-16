import pandas as pd
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import click
from src.generate_sum_statistics import generate_summary_statistics
from src.plot_histograms import plot_histograms
from src.plot_target_distribution import plot_target_distribution
from src.plot_cat_barplots import plot_categorical_stacked_barplots
from src.plot_numeric_boxplots import plot_numeric_boxplots
from src.setup_logger import setup_logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@click.command()
@click.option('--input_data_path', type=click.Path(exists=True), required=True, help='Path to the input CSV data file')
@click.option('--output_prefix', type=click.Path(), required=True, help='Output directory prefix')
def perform_eda(input_data_path, output_prefix):
    """
    Main EDA driver function.
    """
    logger = setup_logger(os.path.basename(__file__))

    try:
        # Load the data
        data = pd.read_csv(input_data_path)

        # Define output directories
        figures_dir = os.path.join(output_prefix, "figures")
        tables_dir = os.path.join(output_prefix, "tables")

        # Ensure the directories exist
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(tables_dir, exist_ok=True)

        # Generate summary statistics
        generate_summary_statistics(data, tables_dir)

        # Generate plots
        plot_histograms(data, figures_dir)
        plot_target_distribution(data, figures_dir)
        plot_categorical_stacked_barplots(data, figures_dir)
        plot_numeric_boxplots(data, figures_dir)

        logger.info("EDA completed. Summary statistics and plots saved.")

    except Exception as e:
        logger.exception("An error occurred during EDA: %s", e)
        raise


if __name__ == "__main__":
    perform_eda()
