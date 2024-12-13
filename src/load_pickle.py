import os
import pickle


def load_pickle(file_path):
    """
    Load a pipeline or model object from a given file path.

    Parameters
    ----------
    file_path : str
        The path to the pipeline object.

    Returns
    ----------
    object
        The trained pipeline object.

    Raises
    ----------
    FileNotFoundError
        If the directory or file does not exist.
    ValueError
        If the filename doesn't end with '.pickle'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("The directory or file does not exist.")
    if not file_path.endswith(".pickle"):
        raise ValueError("Filename must end with '.pickle'.")

    with open(file_path, 'rb') as f:
        return pickle.load(f)
