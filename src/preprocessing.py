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
    df['clean_content'] = df['content'].apply(clean_content)
    df['tokenized_content'] = df['clean_content'].apply(tokenize_text)
    df['processed_content'] = df['tokenized_content'].apply(remove_punctuation)
    df['processed_content'] = df['processed_content'].apply(lambda x: ' '.join(x))
    df['content_len'] = df['processed_content'].apply(len)
    df['content_word'] = df['processed_content'].apply(lambda x: len(x.split()))
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