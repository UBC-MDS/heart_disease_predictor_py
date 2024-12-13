import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_zip import read_zip

def extract(url, path):
    """Downloads data zip data from the web to a local filepath and extracts it."""
    try:
        read_zip(url, path)
    except:
        os.makedirs(path)
        read_zip(url, path)

@click.command()
@click.option('--url', type=str, help="The URL of the dataset to be downloaded")
@click.option('--path', type=str, help="The path to directory to write the raw data to")
def main(url, path):
    extract(url, path)

if __name__ == '__main__':
    main()