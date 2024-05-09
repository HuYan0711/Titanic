import pandas as pd
import matplotlib.pyplot as plt
from pandasql import sqldf
from data_preprocess import feature_encoding

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set font
plt.rcParams['axes.unicode_minus'] = False  # Solve problem that '-' showed as squre

'''
1.This file is used to analyze data and show the date by figures.
2.Process features which are containing null values or not "int" type 
'''

# data_train = pd.read_csv('./Titanic_data/train.csv')
# data_test = pd.read_csv('./Titanic_data/test.csv')
data_train_xlsx = pd.read_excel('./Titanic_data/train_filled.xlsx')
data_test_xlsx = pd.read_excel('./Titanic_data/test.xlsx')


def plot_bar(data, fig_name):
    fig = plt.figure()
    # Create stack shape bar figure
    df = pd.DataFrame(data).transpose()
    df.plot(kind='bar', stacked=True)
    # plt.xlabel(u"Cabin")
    # plt.ylabel(u"Amount of survived people ")
    plt.show()
    plt.savefig(fig_name, dpi=300)


def cabin_survive_statistics():
    query = """
        SELECT Cabin, Survived
        FROM data_train
        WHERE Cabin IS NOT NULL
    """
    # 使用 sqldf 函数执行 SQL 查询
    cols = sqldf(query, locals())
    cabin = {}
    cabin_survive = {}
    # cabin_nosurvive={}
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


# cabin_survive={'C': 35, 'G': 2, 'D': 25, 'A': 7, 'B': 35, 'F': 8, 'E': 24, 'T': 0}
# cabin_nosurvive={'C': 24, 'E': 8, 'G': 2, 'D': 8, 'A': 8, 'B': 12, 'F': 5, 'T': 1}
# data={'C': [35,24], 'G': [2,2], 'D': [25,8], 'A': [7,8], 'B': [35,12], 'F': [8,5], 'E': [24,8], 'T': [0,1]}

def fillna_by_median(xlsx_path, column_name: str):
    # Fill null value with median value of data, like 'age'
    df = pd.read_excel(xlsx_path)
    median_number = df[column_name].median()
    df[column_name] = df[column_name].fillna(median_number)
    df.to_excel(xlsx_path, index=False)


def extract_cabin_first_letter(cabin_value):
    # Consider that 'Cabin' column contains several cabins: 'C', 'G', 'D', 'A', 'B', 'F', 'E', 'T',if value is null,fill it with 'Z'
    if pd.isnull(cabin_value):
        return 'Z'
    return str(cabin_value)[0] if isinstance(cabin_value, str) else 'Z'


def add_cabin_letter(df):
    df['Cabin_Letter'] = df['Cabin'].apply(extract_cabin_first_letter)

    return df


def cabin_to_letter(xlsx_path):
    df = pd.read_excel(xlsx_path)
    # Add 'Cabin_Letter' column
    processed_data = add_cabin_letter(df['Cabin'])

    processed_data.to_excel(xlsx_path, index=False)


def feature_encoding(xlsx_path, columns: list):
    # values->numbers
    df = pd.read_excel(xlsx_path)
    for column in columns:
        feature_map = {unique_value: idx for idx, unique_value in enumerate(df[column].unique())}
        # Apply "label coding"
        df[column] = df[column].map(feature_map)
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
