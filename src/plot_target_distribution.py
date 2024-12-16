import os
import matplotlib.pyplot as plt

def plot_target_distribution(data, figures_dir):
    """
    Plots the distribution of the target variable and saves the figure.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - figures_dir (str): Directory where the target variable distribution figure will be saved.
    """
    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "eda_output_target_distribution.png")

    if data.empty or 'num' not in data.columns:
        raise ValueError("Input DataFrame is empty")

    plt.figure(figsize=(8, 6))
    data['num'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Distribution of Target Variable (num)')
    plt.xlabel('Class (num)')
    plt.ylabel('Count')
    plt.savefig(output_path)
    plt.close()
    print(f"Target distribution plot saved to {output_path}")
