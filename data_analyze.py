import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font
plt.rcParams['axes.unicode_minus'] = False  # Solve problem that '-' showed as squre

'''
1.This file is used to analyze data and show the date by figures.
2.Process features which are containing null values or not "int" type 
'''

def plot_bar(data, fig_name):

    fig = plt.figure()
    df = pd.DataFrame(data).transpose()
    df.plot(kind='bar', stacked=True)
    plt.show()
    plt.savefig(fig_name, dpi=300)

    return 0


def cabin_survive_statistics():
    query = """
        SELECT Cabin, Survived
        FROM data_train
        WHERE Cabin IS NOT NULL
    """
    cols = sqldf(query, locals())
    cabin = {}
    cabin_survive = {}
    for index, row in cols.iterrows():
        if cabin_value is not None and str(cabin_value):
            letter = str(cabin_value)[0]
            if letter not in cabin:
                cabin[letter] = 1
            else:
                cabin[letter] = cabin[letter] + 1

            if letter not in cabin_survive and row['Survived'] == 1:
                cabin_survive[letter] = 1
            elif letter in cabin_survive and row['Survived'] == 1:
                cabin_survive[letter] += 1
    return cabin, cabin_survive


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
