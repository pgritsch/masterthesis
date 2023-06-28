import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import sklearn.metrics as s_m


def encode_to_int(col: pd.Series) -> int:
    return preprocessing.LabelEncoder().fit_transform(col)


def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]


def mean_percent_deviation_error(true: np.ndarray, predicted: np.ndarray) -> float:
    return np.mean(np.abs(true - predicted) / true)


def get_src_dir():
    return os.path.dirname(os.path.realpath(__file__))


def get_project_dir():
    return os.path.dirname(get_src_dir())


def get_data_dir():
    return os.path.join(get_project_dir(), "data")

def print_metrics(y_test, predictions):
    rmse = np.sqrt(s_m.mean_squared_error(y_test, predictions))
    mae = s_m.mean_absolute_error(y_test, predictions)
    mape = s_m.mean_absolute_percentage_error(y_test, predictions)
    print('RMSE: %.3f \nMAE: %.3f \nMAPE: %.3f' % (rmse, mae, mape))
