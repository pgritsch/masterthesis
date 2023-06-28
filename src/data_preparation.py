import math
import os

import numpy as np
from sklearn import preprocessing

from util import *


def load_dataset() -> pd.DataFrame:
    return pd.read_excel(os.path.join(get_data_dir(), 'dataset.xlsx'))


def retrieve_region(x: dict, region_key: str, provider_key: str) -> str:
    fid = x[region_key]
    provider = x[provider_key]

    if provider == 'GCP':
        if 'europe-west1' in fid:
            return 'centralEurope'
        elif 'europe-west2' in fid:
            return 'northernEurope'
        elif 'us-east4' in fid:
            return 'northVirginia'
    elif provider == 'AWS':
        if 'eu-central-1' in fid:
            return 'centralEurope'
        elif 'eu-west-2' in fid:
            return 'northernEurope'
        elif 'us-east-1' in fid:
            return 'northVirginia'

    raise KeyError(provider + ' ' + fid + ' are unhandled')


def get_storage_related_cols() -> list:
    return ['numberDownloadFiles', 'sizeDownloadInMB', 'numberUploadFiles', 'sizeUploadInMB', 'storageRegionEnc',
            'storageProviderEnc']


def get_concurrency_related_cols() -> list:
    return ['loopCounter', 'maxLoopCounter']


def get_function_related_cols() -> list:
    return ['functionRegionEnc', 'functionProviderEnc', 'functionNameEnc', 'functionTypeEnc', 'wfTypeEnc']


def get_time_related_cols() -> list:
    return ['dayofweek', 'timeofday']


def retrieve_function_region(x: dict) -> str:
    return retrieve_region(x, 'functionId', 'functionProvider')


def retrieve_storage_region(x: dict) -> str:
    return retrieve_region(x, 'storageRegion', 'storageProvider')


def retrieve_wf_type(x: dict) -> str:
    fName = x['functionName']
    if 'genome' in fName:
        return 'genome'
    elif 'bwa' in fName:
        return 'bwa'

    raise KeyError(fName + ' unsupported')


def train_test_split_with_criterion(func, dataframe: pd.DataFrame, input_cols: list, output_col, group_col) -> tuple:
    df_train = dataframe[dataframe.apply(lambda x: not func(x), axis=1)]
    df_test = dataframe[dataframe.apply(lambda x: func(x), axis=1)]

    get_idx = lambda x: x if type(x) == 'int' else dataframe.columns.get_loc(x)

    input_cols = [get_idx(c) for c in input_cols]
    output_col = get_idx(output_col)
    group_col = get_idx(group_col)

    X_train = df_train.iloc[:, input_cols].values
    y_train = df_train.iloc[:, output_col].values
    group_train = encode_to_int(df_train.iloc[:, group_col])

    X_test = df_test.iloc[:, input_cols].values
    y_test = df_test.iloc[:, output_col].values

    return X_train, y_train, group_train, X_test, y_test, df_train, df_test


def load_preprocessed_dataset(force_reload=False,
                              filename='processed_dataset.pkl',
                              remove_duplicates=False) -> pd.DataFrame:
    filepath = os.path.join(get_data_dir(), filename)
    if not force_reload and os.path.exists(filepath):
        df = pd.read_pickle(filepath)
        if remove_duplicates:
            df = df.drop_duplicates()
        return df

    df = load_dataset()

    df = df[df['Event'] == 'FUNCTION_END']
    df = df.iloc[:, [2, 3, 4, 6, 7, 10, 13, 14, 15, 16, 18, 19, 22, 23, 24, 25]]

    df['functionProvider'] = df['functionId'].map(lambda fid: 'AWS' if 'arn:aws' in fid else 'GCP')

    df['functionRegion'] = df.apply(retrieve_function_region, axis=1)
    df['wfType'] = df.apply(retrieve_wf_type, axis=1)

    df['storageRegion'] = df.apply(retrieve_storage_region, axis=1)

    regions = set(df['functionRegion']).union(set(df['storageRegion']))
    region_encoder = preprocessing.LabelEncoder().fit(list(regions))
    df['functionRegionEnc'] = region_encoder.transform(df['functionRegion'])
    df['storageRegionEnc'] = region_encoder.transform(df['storageRegion'])

    providers = set(df['functionProvider']).union(set(df['storageProvider']))
    provider_encoder = preprocessing.LabelEncoder().fit(list(providers))
    df['functionProviderEnc'] = provider_encoder.transform(df['functionProvider'])
    df['storageProviderEnc'] = provider_encoder.transform(df['storageProvider'])

    df['functionNameEnc'] = encode_to_int(df['functionName'])
    df['functionTypeEnc'] = encode_to_int(df['functionType'])
    df['wfTypeEnc'] = encode_to_int(df['wfType'])

    df['upAll'] = df['upAll'].map(lambda x: x if not math.isnan(x) else 0)
    df['downAll'] = df['downAll'].map(lambda x: x if not math.isnan(x) else 0)

    df['startTime'] = pd.to_datetime(df['startTime'])
    df['dayofweek'] = df['startTime'].dt.weekday
    df['timeofday'] = df['startTime'].dt.hour * 60 * 60 + df['startTime'].dt.minute * 60 + df['startTime'].dt.second

    df['RTT'] = df['RTT'] / 1000
    df['upAll'] = df['upAll'] / 1000
    df['downAll'] = df['downAll'] / 1000

    df['ct'] = df['RTT'] - df['upAll'] - df['downAll']
    df['datatransferTime'] = df['RTT'] - df['ct']

    df['kFoldGroupEnc'] = df['functionType'] + ':' + df['functionProvider'] + ':' + df['functionRegion']
    df['kFoldGroupEnc'] = encode_to_int(df['kFoldGroupEnc'])

    df.sort_values(by=['kFoldGroupEnc'], inplace=True)

    df.to_pickle(filepath)

    if remove_duplicates:
        df = df.drop_duplicates()
    return df


class CousinCrossValidation:

    @classmethod
    def split(cls,
              X: pd.DataFrame,
              y: np.ndarray = None,
              groups: np.ndarray = None):

        """Returns to a grouped time series split generator."""
        assert len(X) == len(groups), (
            "Length of the predictors is not"
            "matching with the groups.")
        # The min max index must be sorted in the range
        for group_idx in range(groups.min(), groups.max()):
            training_group = group_idx
            # Gets the next group right after
            # the training as test
            test_group = group_idx + 1
            training_indices = np.where(
                groups == training_group)[0]
            test_indices = np.where(groups == test_group)[0]
            if len(test_indices) > 0:
                # Yielding to training and testing indices
                # for cross-validation generator
                yield training_indices, test_indices
