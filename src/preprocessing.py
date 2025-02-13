import pandas as pd
from src.text_cleaning import clean_content
from src.text_tokenization import tokenize_text, remove_punctuation

def prepreprocess_data(data):
    df = data.copy()
    df['content_len'] = df['content'].apply(len)
    df['content_word'] = data['content'].apply(lambda x: len(x.split()))
    return df

def preprocess_data(data):
    df = data.copy()
    df['content'] = df['content'].apply(lambda x: ' '.join(remove_punctuation(tokenize_text(clean_content(x)))))
    return df

def get_dataframe_info(df):
    info_dict = {
        'num_rows': len(df),
        'num_columns': len(df.columns),
        'column_names': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'non_null_counts': df.count().to_dict()
    }
    return info_dict