import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import click
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.setup_logger import setup_logger


# Define function to perform EDA
def perform_eda(input_data_path, output_prefix):
    # Load the cleaned and validated dataset
    data = pd.read_csv(input_data_path)

    # Define output directories
    figures_dir = os.path.join(output_prefix, "figures")
    tables_dir = os.path.join(output_prefix, "tables")

    # Ensure the directories exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Summary statistics
    summary_stats = data.describe()
    summary_stats.to_csv(f"{tables_dir}/eda_output_summary_stats.csv")

    # Histogram for all numeric columns
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    num_features = len(numeric_features)
    cols = 4
    rows = (num_features + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        axes[i].hist(data[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axes[i].set_title(f'Distribution of {col}', fontsize=10)
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel('Count', fontsize=8)

    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig(f"{figures_dir}/eda_output_raw_feature_distributions.png", dpi=300)
    plt.close()

    # Plot target variable distribution
    plt.figure(figsize=(8, 6))
    data['num'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable (num)')
    plt.xlabel('Class (num)')
    plt.ylabel('Count')
    plt.savefig(f"{figures_dir}/eda_output_target_distribution.png")
    plt.close()

    # Grouped stacked bar plots for categorical features
    categorical_features = ['cp', 'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang']
    fig, axes = plt.subplots(4, 2, figsize=(16, 16), constrained_layout=True)
    axes = axes.flatten()

    for i, feature in enumerate(categorical_features):
        counts = data.groupby(['num', feature]).size().unstack(fill_value=0)
        counts.plot(kind='bar', stacked=True, ax=axes[i], colormap='tab10', edgecolor='black')
        axes[i].set_title(f'{feature} by Class (num)')
        axes[i].set_xlabel('Class (num)')
        axes[i].set_ylabel('Count')

    for i in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[i])

    plt.savefig(f"{figures_dir}/eda_output_categorical_stacked_barplots.png")
    plt.close()

    # Grouped boxplots for numeric features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = numeric_features.drop('num', errors='ignore')  # Exclude target variable
    num_features = len(numeric_features)
    cols = 2
    rows = (num_features + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows), constrained_layout=True)
    axes = axes.flatten()
    for i, feature in enumerate(numeric_features):
        data.boxplot(column=feature, by='num', ax=axes[i], grid=False, showmeans=True)
        axes[i].set_title(f'{feature} by Class (num)')
        axes[i].set_xlabel('Class (num)')
        axes[i].set_ylabel(feature)
    # Remove empty subplots
    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])
    plt.suptitle('')  # Remove automatic subtitle
    plt.savefig(f"{figures_dir}/eda_output_numeric_boxplots.png")
    plt.close()

    print("EDA completed. Summary statistics and grouped plots are saved.")


# Use click to handle command-line arguments
@click.command()
@click.option('--input_data_path', type=click.Path(exists=True), required=True, help='Path to the input CSV data file')
@click.option('--output_prefix', type=click.Path(), required=True, help='Output file prefix for generated visualizations and tables')
def main(input_data_path, output_prefix):
    logger = setup_logger(os.path.basename(__file__))

    try:
        perform_eda(input_data_path, output_prefix)
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        raise


if __name__ == "__main__":
    main()
