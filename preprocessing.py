import pandas as pd
import typing_extensions
import numpy as np
import os
import re 
from constants import RAWDATASETPATH, ORDINALITY_MAPPING, REGEXES


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
    datadir = "../PythonProjects/post_processed_datasets/"
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    post_processed_dataframe.to_csv(os.path.join(datadir, save_filename))



save_out_dataset(sampled_postprocessed_df, 'sampled_employee_df_25_pct.csv')


# if __name__ == "__main__":
#     df = load_dataset(DATASETPATH)