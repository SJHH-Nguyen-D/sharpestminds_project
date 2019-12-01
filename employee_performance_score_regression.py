import pandas_profiling
import pandas as pd
import numpy as np
from scipy.stats import iqr
from numpy import percentile
from sklearn.preprocessing import LabelEncoder
from missingpy import KNNImputer
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import RandomForestRegressor
from tpot import TPOTRegressor
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)

from constants import *


def get_outliers_and_extremes(df, numeric_attribute='job_performance'):
    from scipy.stats import iqr
    from numpy import percentile

    IQR = iqr(df[numeric_attribute], axis=0, rng=(25, 75), scale='raw', nan_policy='propagate', interpolation='linear', keepdims=False)
    q1 = percentile(df[numeric_attribute], 0.25, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    q3 = percentile(df[numeric_attribute], 0.75, axis=0, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    
    outliers = df[(df[numeric_attribute] <= (q1 - (IQR * 1.5))) | (df[numeric_attribute] <= (q3 + (IQR * 1.5)))]
    extremes = df[(df[numeric_attribute] <= (q1 - (IQR * 1.5))) | (df[numeric_attribute] <= (q3 + (IQR * 1.5)))]
    return outliers


def detect_highly_correlated_variables(dataframe):
    """ Uses the pandas profiler to detect the highly correlated variables to drop """
    import pandas_profiling
    profile = pandas_profiling.ProfileReport(dataframe)
    rejected_variables = profile.get_rejected_variables(threshold=0.9)
    return rejected_variables


def df_by_type_splitter(dataframe):
    """ a larger dataframe into immediately identifiable numeric and other type dataframes"""
    num_df = dataframe._get_numeric_data().copy()
    cat_df = dataframe.select_dtypes(exclude = [int, float]).copy()
    return num_df, cat_df


def binary_variable_mapping(dataframe, mapping_dict):
    """ mapping of numeric values to binary outcomes """
    # yes and no mappings
    yes_no_mapping = {'Yes': 1, 'No': 0}
    for feature in dataframe.columns:
        if "Yes" in dataframe[feature].unique():
            dataframe[feature] = dataframe[feature].replace(yes_no_mapping)

    for feature_name, mapping in mapping_dict.items():
            dataframe[feature_name] = dataframe[feature_name].replace(mapping)


def ordinal_variable_mapping(dataframe, mapping):
    for feats, cat in mapping:
        for att in feats:
            indiv_feat_mapping = {key: val for val, key in enumerate(cat)}
            dataframe[att].replace(to_replace=indiv_feat_mapping, inplace=True)


def nominal_feature_mapping(dataframe):
    """transform mapping for nominal features"""
    from sklearn.preprocessing import LabelEncoder
    nominal_categorical_encoding_manifest = {}
    # temp fill of NaN values with a string
    dataframe.fillna('Null', inplace=True)
    for col in dataframe.columns:
        le = LabelEncoder()
        le.fit(dataframe[col].values.ravel())
        dataframe[col] = le.transform(dataframe[col].values.ravel())
        nominal_categorical_encoding_manifest[col] = list(le.classes_)
        if dataframe[col].isnull().sum() > 0:
            # fill back temp-fill "Null" encoded values with actual NaN values for later imputer
            dataframe[col].replace(to_replace=list(le.classes_).index('Null'), value=np.nan, inplace=True)
            null_index = list(le.classes_).index('Null')
        le = None


def transform_all(dataframe, binary_mapping, ordinal_mapping):
    """all transformations into one function"""
    # binary mappings
    binary_feature_names = list(set([col for col in dataframe.columns if len(dataframe[col].unique()) <= 3]) - set(["v51", "v229", "v13"]))
    binary_df = dataframe[binary_feature_names]
    binary_variable_mapping(binary_df, binary_mapping)
    
    # ordinal mappings
    ordinal_df = dataframe[ORDINAL_FEATURE_NAMES]
    ordinal_variable_mapping(ordinal_df, ordinal_mapping)

    # nominal encoding
    nominal_df = dataframe[NOMINAL_FEATURE_NAMES]
    nominal_feature_mapping(nominal_df)
    
    # combine all
    transformed_dataframe = pd.concat([binary_df, ordinal_df, nominal_df], axis=1)
    return transformed_dataframe


def impute_missing_for_dataframe(dataframe, target='job_performance'):
    """ The imputer function should be used on a dataframe that has already been numerically encoded """
    from missingpy import KNNImputer #, MissForest
    
    X = dataframe.loc[:, dataframe.columns != target].values
    y = dataframe[target].values

    # imputer object
    knn = KNNImputer(n_neighbors=5, 
                    weights="uniform",
                    metric="masked_euclidean",
                    row_max_missing=0.8,
                    col_max_missing=0.8, 
                    copy=True)
    knn_missing_imputation = knn.fit_transform(X)
    imputed_dataframe = pd.DataFrame(knn_missing_imputation, 
                                     columns = dataframe.columns[dataframe.columns != target])
    imputed_dataframe[target] = pd.Series(y)
    return imputed_dataframe


def round_selected_attributes_imputed(dataframe_to_round, dataframe_not_round):
    rounded_dataframe = dataframe_to_round.apply(lambda x: x.round())
    dataframe = pd.concat([rounded_dataframe, dataframe_not_round], axis=1).reset_index()
    # dataframe.drop("index", axis=1, inplace=True)
    return dataframe


def split_dataframe(dataframe, target="job_performance", test_size=0.3, random_state=123):
    from sklearn.model_selection import train_test_split
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.loc[:, dataframe.columns != target].values,
        dataframe[target].values.ravel(),
        test_size=test_size,
        random_state=random_state)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test


def select_n_features(X, Y, n_features=10):
    """ uses the mlxtend module to select a number of features to keep in the dataframe """
    from mlxtend.feature_selection import SequentialFeatureSelector as SFS
    from sklearn.ensemble import RandomForestRegressor

    # # Build RF regressor to use in feature selection
    rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)

    sfs = SFS(rfr, 
              k_features=n_features, 
              forward=True, 
              floating=False, 
              scoring='r2',
              n_jobs=-1,
              cv=10)

    sfs = sfs.fit(X, Y)

    feature_indices = sfs.k_feature_idx_
    feature_names = sfs.k_feature_names_

    return feature_indices, feature_names

def univariate_feature_selection_with_GUS(dataframe):
    """ uses univariate statistics such as mutual information regression to select the k best features
    for a regression problem """
    from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_regression

    X, y = xy_split(df)
    start = default_timer()
    select_features_gus = GenericUnivariateSelect(score_func=mutual_info_regression, mode="k_best", param=round((df.shape[1]-1)/3)).fit_transform(X, y)
    end = default_timer()
    print("Elapsed Time for feature selection: {}s".format(end-start))
    print(GenericUnivariateSelect.scores_)
    return select_features_gus


# Train/test split using the new 10 feature selected dataset
def model_selection_and_HPO(dataframe, target="job_performance", test_size=0.25, r_seed=123):
    """ Pass in the dataframe that has gone through feature selection
    Uses the TPOT regressor module from TPOT to perform MS and HPO. As this modeling uses some element
    of stochasticity, it may provide different results every time. The longer you run this,
    the more similar the final models will look like in the end.
    
    Finally outputs a .py file with the selected model and its hyperparameters, for which we can import.
    """
    import TPOT 
    from sklearn.model_selection import train_test_split
    import timeit
    from tpot import TPOTRegressor
    from sklearn.metrics import (
        confusion_matrix,
        roc_auc_score,
        precision_recall_fscore_support,
        accuracy_score,
    )

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.loc[:, dataframe.columns != target].values,
        dataframe[target].values.ravel(),
        test_size=test_size,
        random_state=r_seed)
    
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # model selection and hyperparameter optimization with TPOT Regressor
    tpot_regressor = TPOTRegressor(generations=20, 
                                   population_size=50, 
                                   cv=10,
                                   random_state=r_seed, 
                                   verbosity=2, 
                                   memory='auto')
    
    start_time = timeit.default_timer()
    tpot_regressor.fit(X_train, y_train)
    y_pred = tpot_regressor.predict(X_test)
    end_time = timeit.default_timer()

    print(f"Total runtime for the Employee dataset: {end_time-start_time}s")
    print("TPOT Score: {}".format(tpot_regressor.score(X_test, y_test)))

    tpot_regressor.export('tpot_exported_pipeline.py')


def main():
    from timeit import default_timer

    # drive.mount(DRIVENAME)
    df = pd.read_csv(FILENAME, header='infer')
    df = (df.drop(index=get_outliers_and_extremes(df, numeric_attribute='job_performance').index, inplace=False, axis=0) # drop outliers
            .drop(labels=detect_highly_correlated_variables(df), inplace=False, axis=1) # drop highly related correlated variables
            .replace(to_replace=CONSIDERED_MISSING_VALUES, value=np.nan, inplace=False) # missing value encoding
            .drop(labels=list(set([feature for feature in df.columns if (df[feature].isnull().sum(axis=0) / df.shape[0]) >= 0.6])-set(detect_highly_correlated_variables(df))),
                 inplace=False, 
                 axis=1))
    df = (df.drop(index=list(df[((df.isnull().sum(axis=1)/df.shape[1]) >= 0.40) == True].index), # drop rows that have more than 40% of values missing
                 inplace=False,
                 axis=0) # drop observations with greater than 40% of values missing
           .drop(labels=[feature for feature in REDUNDANT_FEATURES if feature in df.columns], inplace=False, axis=1)) # drop redundant features

    numeric_df, categorical_df = df_by_type_splitter(df)
    categorical_df = transform_all(categorical_df, BINARY_VARIABLE_MAPPING, ORDINAL_VARIABLE_MAPPING)

    df = (pd.concat([numeric_df, categorical_df], axis=1)
            .drop(labels=numeric_df.loc[:, ((numeric_df.isnull().sum(axis=0) / numeric_df.shape[0]) >= 0.6)].columns, 
                  inplace=False, axis=1))

    # Imputation of Missing Features
    start_time = default_timer()
    df = impute_missing_for_dataframe(df, target="job_performance")
    df = round_selected_attributes_imputed(df[categorical_df.columns], df[[col for col in df.columns if col not in categorical_df.columns]])
    end_time = default_timer()
    print("Impute time: {}".format(end_time - start_time))

    # Feature Selection
    start_time = default_timer()
    X_train, X_test, y_train, y_test = split_dataframe(df, target="job_performance", test_size=0.3, random_state=123)
    feature_indices, feature_names = select_n_features(X_train, y_train, n_features=round(0.33*len(df.columns)-1))
    end_time = default_timer()
    print("Feature Selection Time: {}".format(end_time - start_time))

    # Model Selection
    start_time = default_timer()
    model_selection_and_HPO(df[feature_names], test_size=0.3)
    end_time = default_timer()
    print("Model Selection Time: {}".format(end_time - start_time))


if __name__ == "__main__":
    main()