import os

def generate_summary_statistics(data, tables_dir):
    """
    Generates summary statistics and saves them as a CSV file.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - tables_dir (str): Directory where the summary statistics CSV will be saved.
    """
    if data.empty:
        raise ValueError("DataFrame is empty")

    summary_stats = data.describe()
    os.makedirs(tables_dir, exist_ok=True)
    output_path = os.path.join(tables_dir, "eda_output_summary_stats.csv")
    summary_stats.to_csv(output_path)
    print(f"Summary statistics saved to {output_path}")
