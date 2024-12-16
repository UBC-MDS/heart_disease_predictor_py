import os
import matplotlib.pyplot as plt

def plot_histograms(data, figures_dir):
    """
    Plots histograms for all numeric columns and saves the figure.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - figures_dir (str): Directory where the histogram figure will be saved.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")

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

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "eda_output_raw_feature_distributions.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Histograms saved to {output_path}")
