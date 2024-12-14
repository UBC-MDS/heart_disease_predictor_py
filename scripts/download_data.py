import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_zip import read_zip
from src.setup_logger import setup_logger


@click.command()
@click.option('--url', type=str, help="The URL of the dataset to be downloaded")
@click.option('--path', type=str, help="The path to directory to write the raw data to")
def main(url, path):
    logger = setup_logger(os.path.basename(__file__))
    try:
        read_zip(url, path)
    except FileNotFoundError:
        os.makedirs(path)
        read_zip(url, path)
    except Exception as e:
        logger.exception("An error occurred: %s", e)
        raise


if __name__ == '__main__':
    main()
