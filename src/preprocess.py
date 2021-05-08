"""
Functions that assist in data preprocessing
"""

import re
import pandas as pd
import numpy as np
import tensorflow as tf
from pandas_profiling import ProfileReport


def change_column_names(columns:list) -> list:
    """
    Function that changes column names by removing special characters

    Args:
        columns (list): List of current column names
    
    Returns:
        list: List of values for the columns where each element has been cleaned
              and fit for purpose
    """
    # Remove special characters
    reg_expr = '[^a-zA-Z0-9 \n\.]'

    # Trim, Lower & Replace Spaces with '_'
    new_columns = ["_".join(re.sub(reg_expr,' ', x).lower().strip().split(' ')) 
                   for x in columns]

    return new_columns


def column_selection(data:pd.core.frame.DataFrame,columns:list = None, na_columns:list = None) -> pd.core.frame.DataFrame:
    """
    Function to get the required columns from the dataframe and drop unwanted columns

    Args:
        data(pd.DataFrame): The dataset to analyse
        columns(list): The list of columns to select from the dataframe
        na_columns(list): The list of columns to drop nas
    
    Returns:
        pd.core.frame.DataFrame: The subselected dataframe
    """
    COLS = ["product", "sub_product", "issue", "sub_issue", "consumer_complaint_narrative", 
            "company", "state", "zip_code", "company_response_to_consumer", 
            "timely_response", "consumer_disputed"]
    if columns is None:
        columns = COLS
    
    if na_columns is None:
        na_columns = ['consumer_complaint_narrative','consumer_disputed']

    return data[columns].dropna(subset = na_columns)

def generate_pandas_profile(data:pd.core.frame.DataFrame, location:str) -> bool:
    """
    Function that runs pandas profiling 

    Args:
        data(pd.DataFrame): The dataset to analyse
        location(str): The path of the location where to store the file
    
    Returns:
        None: Stores the file in a given location
    """
    if location[-5:] not in [".html",".json"]:
        location = location+".json"

    profile = ProfileReport(data, title="Pandas Profiling Report")
    profile.to_file(location)

    return True

def change_zipcode_col(data:pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Function that processes the zipcode column

    Args:
        data(pd.DataFrame): The dataset to analyse with a column called 'zip_code'

    Returns:
        data(pd.DataFrame): The dataset

    Raises
        Exception: if 'zip_code' is not in Data
    """
    # Validate Column
    if 'zip_code' not in list(data.columns):
        raise Exception("Column zip_code is not found")

    data['zip_code'] = data['zip_code'].str.replace('X', '0', regex=True)
    data['zip_code'] = data['zip_code'].str.replace(r'\[|\*|\+|\-|`|\.|\ |\$|\/|!|\(', '0', regex=True)
    data['zip_code'] = data['zip_code'].fillna(0)
    data['zip_code'] = data['zip_code'].astype('int32')
    data['zip_code'] = data['zip_code'].apply(lambda x: x//10000)

    return data
    

def get_one_hot_vector(data:pd.core.frame.DataFrame, 
                       cols:list, validate:bool = False ) -> list:
    """
    Function that gets the one hot code values for the given columns in a dataframe

    Args:
        data(pd.DataFrame): The dataset to analyse
        cols(list): The list of columns that required one-hot encoding
        validate(bool): Flag to indicate if assertion is to be done

    Returns:
        list: List of arrays with encoded values

    Raises:
        None
    """
    one_hot_x = [np.asarray(tf.keras.utils.to_categorical(data[feature_name].values)) 
                 for feature_name in data[cols]]
    

    return one_hot_x

def get_text_embedding(data:pd.core.frame.DataFrame, 
                       cols:list, validate:bool = False) -> list:
    """
    Function that gets the input to text embeddings values for the given columns in a dataframe

    Args:
        data(pd.DataFrame): The dataset to analyse
        cols(list): The list of columns that required one-hot encoding
        validate(bool): Flag to indicate if assertion is to be done

    Returns:
        list: List of arrays with encoded values

    Raises:
        None
    """
    embedding_x = [np.asarray(data[feature_name].values).reshape(-1) 
                   for feature_name in data[cols]]
    
    return embedding_x

