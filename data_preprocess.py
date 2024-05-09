import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from data_analyze import cabin_to_letter, fillna_by_median, feature_encoding

'''
NOTICE:
1.Elements in TensorFlow vector must be the same data type 
2.Each components of datas input to nn must be numbers ,features like 'sex' is str type ,must be converted to numbers,like 'one-hot',male as 0 ;female as 1
'''


def features2tensor(file_path: str, column_list: list):
    # column_list like: ['feature1', 'feature2', ..., 'featureN']
    df = pd.read_excel(file_path)
    features = df[column_list]
    # pandas DataFrame->numpy list
    features_array = features.values
    # Standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features_array)
    features_tensor = torch.tensor(scaled, dtype=torch.float32)
    return features_tensor



fillna_by_median('./Titanic_data/train.csv', './Titanic_data/train_filled.xlsx', 'Age')
fillna_by_median('./Titanic_data/test.csv', './Titanic_data/test_filled.xlsx', 'Age')

cabin_to_letter('./Titanic_data/train_filled.xlsx')
cabin_to_letter('./Titanic_data/test_filled.xlsx')

feature_encoding('./Titanic_data/train_filled.xlsx', ['Sex', 'Embarked', 'Cabin_Letter'])
feature_encoding('./Titanic_data/test_filled.xlsx', ['Sex', 'Embarked', 'Cabin_Letter'])

df = pd.read_excel('./Titanic_data/train_filled.xlsx')

numpy_labels = df['Survived'].to_numpy()  # 使用to_numpy()明确转换为numpy数组
# 将numpy数组转换为PyTorch tensor
train_labels = torch.tensor(numpy_labels, dtype=torch.float32)

train_features = features2tensor('./Titanic_data/train_filled.xlsx',
                                 ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Letter'])
test_features = features2tensor('./Titanic_data/test_filled.xlsx',
                                ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Cabin_Letter'])

X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(test_features)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
