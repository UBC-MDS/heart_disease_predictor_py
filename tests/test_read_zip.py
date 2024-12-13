import os
import sys
import pytest
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.read_zip import read_zip

# make empty directory for data files to be downloaded to
if not os.path.exists('tests/test_zip_empty'):
    os.makedirs('tests/test_zip_empty')

# make directory with a file for data files to be downloaded to
if not os.path.exists('tests/test_zip_file'):
    os.makedirs('tests/test_zip_file')
with open('tests/test_zip_file/test.txt', 'w') as file:
    pass  # The 'pass' statement does nothing, creating an empty file

test_files_txt_csv = ['test1.txt', 'test2.csv']
test_files_txt_csv_nonempty = ['test1.txt', 'test2.csv', 'test.txt']
# URL for cases (zip file containing 'test1.txt' and 'test2.csv')
url_txt_csv_zip = 'https://github.com/UBC-MDS/heart_disease_predictor_py/raw/main/tests/files_txt_csv.zip'

# test read_zip function can download and extract a zip file containing two files of two different types 
def test_read_zip_txt_csv():
    read_zip(url_txt_csv_zip, 'tests/test_zip_empty')
    # List of files you expect to find in the directory
    for file in test_files_txt_csv:
        file_path = os.path.join('tests/test_zip_empty', file)
        assert os.path.isfile(file_path)
    # clean up unzipped files
    for file in test_files_txt_csv:
        if os.path.exists(file):
            os.remove(file)

# test read_zip function can download and extract a zip file containing two files of two different types
# into a directory that already contains a file
def test_read_zip_txt_csv_nonempty():
    read_zip(url_txt_csv_zip, 'tests/test_zip_file')
    # List of files you expect to find in the directory
    for file in test_files_txt_csv_nonempty:
        file_path = os.path.join('tests/test_zip_file', file)
        assert os.path.isfile(file_path)
    # clean up unzipped files
    for file in test_files_txt_csv:
        if os.path.exists(file):
            os.remove(file)
