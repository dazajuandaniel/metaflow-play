"""
Unit Tests for Utils.py
"""
import sys
import os
from os import listdir
from os.path import isfile, join
sys.path.append(os.path.join(os.getcwd(), "src"))
import pytest

import pandas as pd
import numpy as np
from preprocess import change_column_names, column_selection, generate_pandas_profile

# Set up DataFrames
# Change Column Name DataFrame
CCN_DF = pd.DataFrame({'special_char!':[1,2,3],"space in between":[1,2,3]})

# Column Selection DataFrame
test_dict = {'product': ['Debt collection', 'Credit card', 'Debt collection'],
             'sub_product': ['I do not know', np.nan, 'Credit card'],
             'issue': ['Disclosure verification of debt','Other','Taking/threatening an illegal action'],
             'sub_issue': ['Right to dispute notice not received',np.nan,'Threatened to sue on too old debt'],
             'consumer_complaint_narrative': ["Narrative1",np.nan,"narrative3"],
             'company': ['Midwest Recovery Systems','CITIBANK, N.A.','Financial Credit Service, Inc.'],
             'state': ['VA', 'NC', 'TX'],
             'zip_code': [np.nan, '286XX', np.nan],
             'company_response_to_consumer': ['response1','response2','response3'],
             'timely_response': ['No', 'Yes', 'No'],
             'consumer_disputed': ['Yes', 'No', 'No'],
             'unwanted_column':  [1,2,3]}

CS_DF = pd.DataFrame(test_dict)

class TestChangeColumnName():
    """
    Test the change_column_names function
    """

    def test_is_list_instance(self):
        """
        Tests that the output is a list
        """
        assert isinstance(change_column_names(CCN_DF), list)

    def test_names(self):
        """
        Tests that the name given is in path
        """
        
        test_list = change_column_names(CCN_DF)

        assert test_list == ['special_char','space_in_between']


class TestColumnSelection():
    """
    Test the column_selection function
    """

    def test_is_pandas(self):
        """
        Tests that the output is a tuple
        """
        assert isinstance(column_selection(CS_DF), pd.DataFrame)

    def test_column_selection(self):
        """
        Tests that the tuple is complete
        """
        data = column_selection(CS_DF)
        assert len(data.columns) == 11
        assert 'unwanted_column' not in data.columns

    def test_drop_na(self):
        """
        Test the dropna capability
        """
        data = column_selection(CS_DF)
        data_test_na = column_selection(CS_DF,na_columns = ["zip_code"])
        data_test_select = column_selection(CS_DF,
                                            columns = ["unwanted_column"], 
                                            na_columns = ['unwanted_column'])

        assert data.shape[0] == 2
        assert data_test_na.shape[0] == 1
        assert data_test_select.shape[1] == 1


class TestGeneratePandasProfile():
    """
    Test the column_selection function
    """

    def test_file_is_saved(self):
        """
        Tests that the file was actually saved
        """
        path_location = 'test_report.json'
        generate_pandas_profile(CCN_DF,path_location)
        files = [f for f in listdir('.') if isfile(join('.', f))]
        assert 'test_report.json' in files
        os.remove(path_location)


    def test_file_json_is_correct(self):
        """
        Tests that the json is correct
        """
        import json
        path_location = 'test_report.json'
        generate_pandas_profile(CCN_DF,path_location)
        
        with open(path_location) as f:
            j = json.load(f)
        
        assert j['table']['n_var'] == 2
        assert j['table']['n'] == 3
        assert list(j['variables'].keys()) == ['special_char!', 'space in between']
        os.remove(path_location)



        
