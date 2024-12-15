import os
import matplotlib.pyplot as plt

def plot_categorical_stacked_barplots(data, figures_dir):
    """
    Plots stacked barplots for categorical features grouped by the target variable.

    Parameters:
    - data (pd.DataFrame): Input dataset.
    - figures_dir (str): Directory where the barplots will be saved.
    """
    if data.empty:
        raise ValueError("DataFrame is empty")

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

    os.makedirs(figures_dir, exist_ok=True)
    output_path = os.path.join(figures_dir, "eda_output_categorical_stacked_barplots.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Categorical stacked barplots saved to {output_path}")
