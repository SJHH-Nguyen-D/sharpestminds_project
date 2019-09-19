from dask import dataframe as dd
from dask import array as da 
import numpy as np 
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ColumnSelector, ExhaustiveFeatureSelector
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

datafile = '/home/dennis/PythonProjects/datasets/post-processed-employee-performance-dataset.csv'

def dataloader(filepath):
    df = dd.read_csv(filepath, header='infer').persist()
    return df

def modeling_preparation(dataframe):
    X_train, X_test, y_train, y_test = train_test_split(
    dataframe.loc[:, test.columns != 'job_performance'].values,
    dataframe['job_performance'].values.ravel(),
    test_size=0.25,
    random_state=42)
    y_train = y_train.ravel()
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test


########## FEATURE SELECTION #############

# # Build RF regressor to use in feature selection
rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)

# # Build step forward feature selection


sfs = SFS(rfr, 
          k_features=round(test.shape[1]/10), 
          forward=True, 
          floating=False, 
          scoring='r2',
          n_jobs=-1,
          cv=10)

# sfs1.attrs: {k_feature_idx_, k_feature_names_ , _ , subsets_, custom_feature_names }

#  Perform Feature Selection
start_time = default_timer()
sfs = sfs.fit(X_train, y_train)
sfs_pred = sfs.predict(X_test, y_test)
end_time = default_timer()
print("Job time: {}s".format(end-time - start_time/1000))


if __name__ == "__main__":
    df = dataloader(datafile)


'''

TO START THE PROCESS
--keyname TEXT (key name in EC2 console)
--keypair PATH (Path to the key pair that matches the keyname)
--name TEXT tag name on EC2
--count INTEGER Number of nodes (default 4)
--dask
-nprocs (number of processes per worker, with default == 1)

example in terminal:

dask-ec2 up --keyname sharpest-minds --keypair ~/Downloads/ssh_aws_keys/sharpest-minds.pem --count 9 --nprocs 8


TO CONNECT TO THE DASK CLUSTER

1) ssh into the head node
2) start ipython shell
3) connect to scheduler running on head node


example in terminal:

dec2 ssh
ipython

import s3fs
fs = s3fs.S3FileSystem(anon=True)
fs.ls('sharpest-minds-data')

from distributed import Executor, s3, progress
e = Executor("127.0.0.1:8786")



TO DESTROY

dec2 destroy

# Running it
This is the path to a s3 bucket
df = s3.read_csv('sharpest-minds-data/post-processed-employee-performance-dataset.csv', lazy=False)

progress(df)

df.columns
df.dtypes

%time df.job_performance.sum().compute()


'''