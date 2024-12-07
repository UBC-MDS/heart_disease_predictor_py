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

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = data.corr()
    plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Correlation Heatmap')
    plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
    plt.savefig(f"{output_prefix}_correlation_heatmap.png")
    plt.close()

    # Histogram of all numeric columns
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        plt.figure()
        data[col].plot(kind='hist', bins=30, edgecolor='black')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f"{output_prefix}_{col}_histogram.png")
        plt.close()

    # Boxplot for detecting outliers in each numeric column
    for col in numeric_cols:
        plt.figure()
        data[col].plot(kind='box')
        plt.title(f'Boxplot of {col}')
        plt.savefig(f"{output_prefix}_{col}_boxplot.png")
        plt.close()

    print("EDA completed. Output files are saved in the specified location.")

# Set up argument parser for command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on a given dataset.')
    parser.add_argument('--input_data_path', type=str, required=True, help='Path to the input CSV data file')
    parser.add_argument('--output_prefix', type=str, required=True, help='Output file prefix for generated visualizations and tables')

    args = parser.parse_args()
    perform_eda(args.input_data_path, args.output_prefix)
