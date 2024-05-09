import torch
import pandas as pd
from pandasql import sqldf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def fillna_by_median(csv_path, xlsx_path, column_name: str):
    df = pd.read_csv(csv_path)
    median_number = df[column_name].median()
    df[column_name] = df[column_name].fillna(median_number)
    df.to_excel(xlsx_path, index=False)
    


def extract_cabin_first_letter(cabin_value):
    if pd.isnull(cabin_value):
        return 'Z'
    return str(cabin_value)[0] if isinstance(cabin_value, str) else 'Z'


def cabin_to_letter(xlsx_path):
    df = pd.read_excel(xlsx_path)
    df['Cabin_Letter'] = df['Cabin'].apply(extract_cabin_first_letter)
    processed_data = df
    processed_data.to_excel(xlsx_path, index=False)


def feature_encoding(xlsx_path, columns: list):
    df = pd.read_excel(xlsx_path)
    for column in columns:
        feature_map = {unique_value: idx for idx, unique_value in enumerate(df[column].unique())}
        df[column] = df[column].map(feature_map)
    df.to_excel(xlsx_path, index=False, engine='openpyxl')

def features2tensor(file_path: str, column_list: list):
    df = pd.read_excel(file_path)
    features = df[column_list]
    features_array = features.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_array)
    features_tensor = torch.tensor(scaled, dtype=torch.float32)
    return features_tensor