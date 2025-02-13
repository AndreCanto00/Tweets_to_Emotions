import pytest
import pandas as pd
from src.preprocessing import preprocess_data, get_dataframe_info

def test_preprocess_data():
    test_data = pd.DataFrame({
        'content': ['hello world', 'test tweet'],
        'sentiment': ['happy', 'sad']
    })
    result = preprocess_data(test_data)
    assert 'content' in result.columns
    assert 'sentiment' in result.columns

def test_get_dataframe_info():
    test_df = pd.DataFrame({
        'content': ['hello world', 'test tweet'],
        'sentiment': ['happy', 'sad']
    })
    info = get_dataframe_info(test_df)
    assert info['num_rows'] == 2
    assert info['num_columns'] == 2