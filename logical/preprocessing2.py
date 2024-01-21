import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import os


def is_float(value):
    try:
        float_value = float(value)
        return True
    except ValueError:
        return False


def is_int(value):
    try:
        int_value = int(value)
        return True
    except ValueError:
        return False


def find_type(data):
    if is_int(data[1]):
        if int(data[1]) > 27000:
            return "str_info"
        else:
            return "float"
    else:
        if is_float(data[1]):
            return "float"
        else:
            if data[1] == 'Yes' or data[1] == 'No':
                return "bool"
            else:
                return "str"


def csv_update(data):
    csv_file_name = "./data/Preprocessing_HHCAHPS.csv"
    with open(csv_file_name, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_row = list(data.columns)
        csv_writer.writerow(header_row)
        max_values_count = max(len(data[column]) for column in data.columns)

        for i in range(max_values_count):
            row = []

            for column in data.columns:
                if i < len(data[column]):
                    row.append(data[column].iloc[i])
                else:
                    row.append(None)

            csv_writer.writerow(row)


def column_elimination(data):
    delete_columns = []
    for column in data.columns:
        if find_type(data[column]) == "str" or find_type(data[column]) == "str_info":
            if np.count_nonzero(data[column] == "-") > len(data[column]) * 0.1:
                delete_columns.append(column)
    data = data.drop(columns=delete_columns)
    data = data.drop("CMS Certification Number (CCN)", axis=1)
    return data


def eliminate_row_space(data):
    index_set = set()

    for column in data.columns:
        if (find_type(data[column]) == "str" or find_type(data[column]) == 'bool' or
                find_type(data[column]) == 'str_info'):
            indexes = np.where((data[column] == '-'))[0]
            index_set.update(indexes)

    index_list = list(index_set)
    data = data.drop(index_list)
    data.reset_index(drop=True, inplace=True)
    return data


def eliminate_outliers(data):
    index_li = []
    for column in data.columns:
        if find_type(data[column]) == "float":
            my_new_data = [float(str(i).replace(',', '')) for i in data[column] if str(i) != '-']
            q1 = np.percentile(my_new_data, 25)
            q3 = np.percentile(my_new_data, 75)
            iqr = q3 - q1
            lower_limit = q1 - 1.5 * iqr
            upper_limit = q3 + 1.5 * iqr
            for index, i in enumerate(data[column]):
                if i == "-":
                    pass
                else:
                    x = float(str(i).replace(',', ''))
                    if (x < lower_limit) | (x > upper_limit):
                        index_li.append(index)
    index_set = set(index_li)
    index_list = list(index_set)
    data.drop(index_list, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data


def nan_complete(data):
    for column in data.columns:
        if find_type(data[column]) == "float":
            new_array = [float(str(i).replace(',', '')) for i in data[column] if str(i) != '-']
            mean_column = round(np.mean(new_array), 2)
            data[column] = data[column].apply(lambda x: mean_column if str(x) == "-" else x)
    return data


def bool_complete(data):
    for column in data.columns:
        if find_type(data[column]) == "bool":
            data[column] = data[column].replace({'Yes': '1', 'No': '0'})
    return data


def elimina_linii_cu_valoare(data):
    data = data[data['HHCAHPS Survey Summary Star Rating'] != 3.71].reset_index(drop=True)
    return data


def data_view(data):
    medians = []
    means = []
    for column in data.columns:
        if find_type(data[column]) == 'float' or find_type(data[column]) == 'bool':
            array = [float(str(i).replace(',', '')) for i in data[column]]
            mean_val = np.mean(array)
            median_val = np.median(array)

            print(f"Mean {column}: {mean_val}")
            print(f"Median {column}: {median_val}")

            means.append(mean_val)
            medians.append(median_val)

    plt.figure(figsize=(15, 10))
    columns = [column for column in data.columns if find_type(data[column]) == "float"
               or find_type(data[column]) == "bool"]
    x = np.arange(len(columns))
    bar_width = 0.35

    plt.bar(x, means, width=bar_width, label='Mean', color='blue', alpha=0.7)
    plt.bar(x + bar_width, medians, width=bar_width, label='Median', color='orange', alpha=0.7)

    plt.title('Overall Means and Medians of Columns')
    plt.xlabel('Columns')
    plt.ylabel('Value')
    columns = [column for column in data.columns if find_type(data[column]) == "float"]
    plt.xticks(x + bar_width / 2, columns, rotation=45, ha='right', fontsize=8)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./diagrams2/medii_mediane.png")
    plt.close()


def encode_text(text, model):
    tokens = word_tokenize(text.lower())
    encoded_vector = [model.wv[word] for word in tokens if word in model.wv]
    return encoded_vector


def transform_str_to_numerical_rep(data):
    for column in data.columns:
        if find_type(data[column]) == 'str':
            tokenized_data = [word_tokenize(sentence.lower()) for sentence in data[column]]
            model = Word2Vec(tokenized_data, vector_size=10, window=5, min_count=1, workers=4)
            data[column] = data[column].apply(lambda sentence: str(np.mean(encode_text(sentence, model))))
    return data


def process_value(value):
    if value == "-":
        return None
    else:
        return float(str(value).replace(',', ''))


def convert_to_float(data):
    for column in data.columns:
        if find_type(column) == "str_info":
            array = [float(i) for i in data[column]]
        else:
            array = [process_value(i) for i in data[column]]
        data[column] = array
    return data


def mean(column):
    return np.mean(column)


def covariance(column1, column2):
    mean1 = mean(column1)
    mean2 = mean(column2)
    n = len(column1)
    covar = sum((column1[i] - mean1) * (column2[i] - mean2) for i in range(n))
    return covar / (n - 1)


def std_dev(column):
    mean_val = mean(column)
    n = len(column)
    variance = sum((x - mean_val) ** 2 for x in column) / (n - 1)
    return variance ** 0.5


def correlation(column1, column2):
    covar = covariance(column1, column2)
    std_dev1 = std_dev(column1)
    std_dev2 = std_dev(column2)
    correlation_coefficient = covar / (std_dev1 * std_dev2)
    return correlation_coefficient


def correlation_elimination(data, y):
    del_column = []
    matrix_correlation = []
    for column in data.columns:
        x = data[column]
        correlationn = correlation(x, y)
        matrix_correlation.append(correlationn)
        if abs(correlationn) < 0.5:
            del_column.append(column)

    for index, column in enumerate(data.columns):
        print(f'{column} with correlation value:{matrix_correlation[index]} ')

    data = data.drop(columns=del_column)
    return data


def not_available(df):
    df.replace({'Not Available': '-', np.nan: '-'}, inplace=True)
    return df


def preprocessing2():
    folder_path = "./diagrams2"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    df = pd.read_csv('./data/HHCAHPS_Provider_Jan2024.csv')
    df = not_available(df)
    df = column_elimination(df)
    df = eliminate_row_space(df)
    df = eliminate_outliers(df)
    df = nan_complete(df)
    df = bool_complete(df)

    data_view(df)
    df = transform_str_to_numerical_rep(df)
    df = convert_to_float(df)
    y = df['HHCAHPS Survey Summary Star Rating'].values.astype(float)
    df = correlation_elimination(df, y)
    df = elimina_linii_cu_valoare(df)
    csv_update(df)


