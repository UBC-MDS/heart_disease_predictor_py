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

test_files_txt_csv = ['test.txt', 'test.csv']
# URL for Case 1 (zip file containing 'test1.txt' and 'test2.csv')
url_txt_csv_zip = 'https://github.com/UBC-MDS/heart_disease_predictor_py/raw/main/tests/files_txt_csv.zip