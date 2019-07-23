"""
MLBOX DEMO
"""
# import supporting libraries
from sklearn.datasets import *
import numpy as np 
import mlbox as mlb
import os
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

##################### downloading the dataset #######################

def download_sklearn_datasets(dataset="iris"):
	sklearn_dataset_dir = os.path.join("./datasets", dataset)
	if not os.path.exists(sklearn_dataset_dir):
		os.makedirs(sklearn_dataset_dir)

	if dataset=="iris":
		data = load_iris()
	elif dataset=="wine":
		data = load_wine()
	elif dataset=="boston":
		data = load_boston()

	X = data.data 
	y = data.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
	
	train = pd.DataFrame(
		X_train, 
		columns=data.feature_names
		)
	train["target"] = pd.Series(y_train)

	test = pd.DataFrame(
		X_test, 
		columns=data.feature_names
		)
	test["target"] = pd.Series(y_test)

	train.to_csv(os.path.join(sklearn_dataset_dir,"train.csv"))
	test.to_csv(os.path.join(sklearn_dataset_dir, "test.csv"))
	
download_sklearn_datasets("boston")


############# preprocessing ####################################
from mlbox.preprocessing import *

trainset_path = "./datasets/{}/train.csv".format(dataset)
testset_path = "./datasets/{}/test.csv".format(dataset)

############## preprocess the data #################

# you can provide a list of paths to the train.csv and the test.csv
paths = [trainset_path, testset_path]

# # performing the cleaning operation and creating a cleaned train and test file
data = Reader(sep=",").train_test_split(paths, target_name="target")

############## making inferences with the best model #########

# delete the drifting data betwen train and test sets
data = mlb.preprocessing.Drift_thresholder().fit_transform(data)

best = mlb.optimisation.Optimiser().evaluate(None, data)
mlb.prediction.Predictor().fit_predict(best, data)

############### optimization ########################

# define a search space for the use of the XGBRegressor
space_xgb={
'ne__numerical_strategy'    :{"search":"choice",
                              "space":[0,'mean','median','most_frequent']},
'ne__categorical_strategy'  :{"search":"choice",
                              "space":[np.NaN,"None"]},
'ce__strategy'              :{"search":"choice",
                              "space":['label_encoding','entity_embedding','dummification']},
'fs__strategy'              :{"search":"choice",
                              "space":['l1','variance','rf_feature_importance']},
'fs__threshold'             :{"search":"uniform",
                              "space":[0.01,0.6]},
'est__strategy'             :{"search":"choice",
                              "space":["XGBoost"]},
'est__max_depth'            :{"search":"choice",
                              "space":[3,4,5,6,7]},
'est__learning_rate'        :{"search":"uniform",
                              "space":[0.01,0.1]},
'est__subsample'            :{"search":"uniform",
                              "space":[0.4,0.9]},
'est__reg_alpha'            :{"search":"uniform",
                              "space":[0,10]},
'est__reg_lambda'           :{"search":"uniform",
                              "space":[0,10]},
'est__n_estimators'         :{"search":"choice",
                              "space":[1000,1250,1500]}
}

best_xgb = mlb.optimization.Optimiser(
	scoring="mean_squared_error", n_folds=5
	).optimise(
	space_xgb, data, max_evals=120
	)

# prediction using best xbg model discovered after search
mlb.prediction.Predictor().fit_predict(best_xgb, data)

# have a look at the best parameters of the model
print(best_xgb)