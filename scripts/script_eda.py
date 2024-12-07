# Exploratory Data Analysis (EDA) Script

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

# Define function to perform EDA
def perform_eda(input_data_path, output_prefix):
    # Load the cleaned and validated dataset
    data = pd.read_csv(input_data_path)

    # Set up directories for saving outputs
    output_directory = os.path.dirname(output_prefix)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Summary statistics
    summary_stats = data.describe()
    summary_stats.to_csv(f"{output_prefix}_summary_stats.csv")

    # Histogram for all numeric columns (combined into a single grid plot)
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    num_features = len(numeric_features)
    cols = 4  # Number of columns in the grid
    rows = (num_features + cols - 1) // cols  # Calculate the required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten()

    # Plot histograms for all numeric features
    for i, col in enumerate(numeric_features):
        axes[i].hist(data[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        axes[i].set_title(f'Distribution of {col}', fontsize=10)
        axes[i].set_xlabel(col, fontsize=8)
        axes[i].set_ylabel('Count', fontsize=8)

    # Remove unused subplots
    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_raw_feature_distributions.png", dpi=300)
    plt.close()

    # Plot target variable distribution
    plt.figure(figsize=(8, 6))
    data['num'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable (num)')
    plt.xlabel('Class (num)')
    plt.ylabel('Count')
    plt.savefig(f"{output_prefix}_target_distribution.png")
    plt.close()

    # Grouped stacked bar plots for categorical features
    categorical_features = ['cp', 'restecg', 'slope', 'thal', 'sex', 'fbs', 'exang']
    num_features = len(categorical_features)
    cols = 2
    rows = (num_features + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows), constrained_layout=True)
    axes = axes.flatten()
    for i, feature in enumerate(categorical_features):
        counts = data.groupby(['num', feature]).size().unstack(fill_value=0)
        counts.plot(kind='bar', stacked=True, ax=axes[i], colormap='tab10', edgecolor='black')
        axes[i].set_title(f'{feature} by Class (num)')
        axes[i].set_xlabel('Class (num)')
        axes[i].set_ylabel('Count')
    # Remove empty subplots
    for i in range(len(categorical_features), len(axes)):
        fig.delaxes(axes[i])
    plt.savefig(f"{output_prefix}_categorical_stacked_barplots.png")
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
    plt.savefig(f"{output_prefix}_numeric_boxplots.png")
    plt.close()

    print("EDA completed. Summary statistics and grouped plots are saved.")

# Set up argument parser for command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on a given dataset.')
    parser.add_argument('--input_data_path', type=str, required=True, help='Path to the input CSV data file')
    parser.add_argument('--output_prefix', type=str, required=True, help='Output file prefix for generated visualizations and tables')

    args = parser.parse_args()
    perform_eda(args.input_data_path, args.output_prefix)
