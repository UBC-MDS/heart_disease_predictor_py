import os
import matplotlib.pyplot as plt

def plot_numeric_boxplots(data, figures_dir):
    """
    Plots grouped boxplots for numeric features and saves the figure.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - figures_dir (str): Directory where the boxplots will be saved.
    """
    if data.empty:
        raise ValueError("Input DataFrame is empty")

    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_features = numeric_features.drop('num', errors='ignore')
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

    for i in range(len(numeric_features), len(axes)):
        fig.delaxes(axes[i])

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "eda_output_numeric_boxplots.png")
    plt.suptitle('')
    plt.savefig(output_path)
    plt.close()
    print(f"Numeric boxplots saved to {output_path}")
