import os
import matplotlib.pyplot as plt


# Save feature coefficient plot as figures
def save_feature_coefficients_plot(coef_df, dir_path, filename):
    """
    Save a bar plot of feature coefficients for logistic regression.

    Parameters:
    ----------
    coef_df: pandas.DataFrame
        The DataFrame containing 'Feature' and 'Coefficient' columns.
    dir_path: str
        The directory where the plot will be saved.
    filename: str
        The name of the saved image file.

    Returns
    ----------
    None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    plt.figure(figsize=(12, 8))
    plt.barh(
        coef_df['Feature'],
        coef_df['Coefficient'],
        color=coef_df['Coefficient'].apply(
            lambda x: 'orange' if x > 0 else 'steelblue'
        )
    )
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Feature Coefficients for Logistic Regression')

    plt.tight_layout()
    plt.savefig(f"{dir_path}/{filename}", dpi=300)
    plt.close()
