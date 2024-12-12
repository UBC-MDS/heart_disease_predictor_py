import os
import pickle


def save_pickle(obj, directory, filename):
    """
    Save an object (e.g. preprocessor or pipeline)
    to a pickle file.

    Parameters
    ----------
    obj : object
        The object to be saved.
    directory : str
        Directory to save the object file.
    filename : str
        Name of the pickle file.

    Returns
    ----------
    None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
