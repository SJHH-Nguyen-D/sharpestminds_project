import pandas as pd
import numpy as np
import os
import re 
from constants import RAWDATASETPATH, ORDINALITY_MAPPING, REGEXES, FEATURES_TO_DROP
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def load_dataset(filepath):
    try:
        dataframe = pd.read_csv(filepath, header='infer')
    except IOError:
        print("The .csv file could not be read in as a dataframe")
    return dataframe


df = load_dataset("/path/to/data.csv", header="infer")


class Preprocessor:
    """ Preprocessor class specifically for this dataset """
    def __init__(self):
        self.inplace=True
        self.axis=1
    
    def fit_transform_all(self, X):
        """ fit all transformations done to this dataset, that are specific for this problem"""
        # drop data points based on percentage of missing data
        # drop series based on column names
        # label encode features
        # impute missing values with KNN
        post_processed_dataframe = (
            fit_transform_drop_by_perecent_missing(X)
            .fit_transform_drop_by_named_cols(_, FEATURES_TO_DROP)
            .fit_transform_encoding()
            .missing_value_imputer()
        )
        return post_processed_dataframe
    
    def fit_transform_drop_by_perecent_missing(self, X):
        processed_dataframe = []
        return processed_dataframe

    def fit_transform_drop_by_named_cols(self, X, drop_series):
        processed_dataframe = X.drop(FEATURES_TO_DROP, inplace=self.inplace, axis=self.axis)
        return processed_dataframe

    def fit_transform_encoding(self, X):
        """ perform all the transformations you need for project """


    def missing_value_imputer(self, X):
        processed_dataframe = []
        return processed_dataframe


def save_out_dataset(post_processed_dataframe, save_filename):
    datadir = "../PythonProjects/post_processed_datasets/"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    post_processed_dataframe.to_csv(os.path.join(datadir, save_filename))


if __name__ == "__main__":
    df = load_dataset(RAWDATASETPATH)
    transformed_dataframe = Preprocessor().fit_transform_all(df)
    save_out_dataset(transformed_dataframe, 'sampled_employee_df_25_pct.csv')