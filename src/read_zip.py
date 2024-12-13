import os
import zipfile
import requests

def read_zip(url, path):
    """
    Read a zip file from the given URL and extract its contents to the specified directory.

    Parameters:
    ----------
    url : str
        The URL of the zip file to be read.
    directory : str
        The directory where the contents of the zip file will be extracted.

    Returns:
    -------
    None
    """
    request = requests.get(url)
    filename_from_url = os.path.basename(url)

    # write the zip file to the directory
    path_to_zip_file = os.path.join(path, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)
        
    # extract the zip file to the directory
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)
