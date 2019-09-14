import pandas as pd
import dask.array as da
import dask.dataframe as dd
import typing_extensions
import numpy as np
import os
import re 
from constants import DATASETPATH, ORDINALITY_MAPPING, REGEXES


def load_dataset(filepath):
    assert os.path.exists(filepath), "The filepath that you have specified does not exists"
    try:
        preprocessing_dataset = pd.read_csv(filepath, header='infer')
    except IOError:
        print("The .csv file could not be read in as a dataframe")
    return preprocessing_dataset


class Preprocessor:
    """ Preprocessor class for project """
    def fit_transform_drop(self, X):
        assert isinstance(X, pd.DataFrame), "The input data structure must be a pandas dataframe."
        DROP_FEATURE_DICT = {key: val for key in REGEXES for val in X.columns if re.match(key) != None}
        
        return dataset

        

    def fit_transform_encoding(self, X):
        """ perform all the transformations you need for project """

        # encode 
        try:
            for i in range(dataset):
                print("Hello World!")
        except IOError:
            print("Invalid inputs")

    def fit_imputer(self, X):
        return dataset


def save_out_dataset(post_processed_dataframe, save_filename):
    if not os.path.exists("../PythonProjects/post_processed_datasets/"):
        os.makedirs("../PythonProjects/post_processed_datasets/")
    post_processed_dataframe.to_csv("../PythonProjects/post_processed_datasets/"+save_filename)



save_out_dataset(sampled_postprocessed_df, 'sampled_employee_df_25_pct.csv')


# if __name__ == "__main__":
#     df = load_dataset(DATASETPATH)