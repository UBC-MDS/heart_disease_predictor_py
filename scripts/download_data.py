import pandas as pd
import click
import os
import requests
import zipfile

@click.command()
@click.option('--url', type=str)
@click.option('--path', type=str)
def main(url, path):
    request = requests.get(url)
    filename_from_url = os.path.basename(url)
    
    # write the zip file to the directory
    path_to_zip_file = os.path.join(path, filename_from_url)
    with open(path_to_zip_file, 'wb') as f:
        f.write(request.content)
        
    # extract the zip file to the directory
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path)

if __name__ == '__main__':
    main()