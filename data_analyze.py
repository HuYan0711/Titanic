import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fillna_by_median(csv_path, column_name: str):
    df = pd.read_csv(csv_path)
    median_number = df[column_name].median()
    df[column_name] = df[column_name].fillna(median_number)
    return df


def extract_cabin_first_letter(cabin_value):
    if pd.isnull(cabin_value):
        return 'Z'
    return str(cabin_value)[0] if isinstance(cabin_value, str) else 'Z'


def cabin_to_letter(df):
    df['Cabin_Letter'] = df['Cabin'].apply(
        extract_cabin_first_letter)
    return df


def feature_encoding(df, columns: list):
    for column in columns:
        feature_map = {unique_value: idx for idx, unique_value in enumerate(df[column].unique())}
        df[column] = df[column].map(feature_map)
    return df


def features2tensor(df, column_list: list):
    features = df[column_list]
    features_array = features.values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_array)
    features_tensor = torch.tensor(scaled, dtype=torch.float32)
    return features_tensor
