"""
TPOT DEMO
"""

from sklearn.datasets import *
import timeit
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    accuracy_score,
)
from pprint import pprint

# load dataset
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

tpot = TPOTClassifier(
	generations=5, 
	population_size=50, 
	verbosity=2, 
	n_jobs=-1
	)

prec_rec_fsc_sup = ["precision", "recall", "fscore", "support"]

start_time = timeit.default_timer()
tpot.fit(X_train, y_train)
y_pred = tpot.predict(X_test)
end_time = timeit.default_timer()
runtime = end_time - start_time
print(f"Total runtime: {runtime}s")


print("\nConfusion Matrix\n{}\n".format(confusion_matrix(y_test, y_pred)))

for met, val in zip(prec_rec_fsc_sup, precision_recall_fscore_support(y_test, y_pred)):
	pprint("{}: {}".format(met, val))

print(accuracy_score(y_test, y_pred))

"""
Best Pipelines (Always random due to stochasticity)

# Iris Dataset
1. LinearSVC
Best pipeline: LinearSVC(PolynomialFeatures(input_matrix, degree=2, include_bias=False, interaction_only=False), C=20.0, dual=False, loss=squared_hinge, penalty=l2, tol=0.1)
Total runtime: 45.1908969899996s
Average Accuracy Score: 0.9666666666666667
Best Accuracy Score: 0.9825757575757577

2. 

# Wine Dataset
1. GradientBoostingClassifier
Best pipeline: GradientBoostingClassifier(input_matrix, learning_rate=0.5, max_depth=3, max_features=0.1, min_samples_leaf=1, min_samples_split=4, n_estimators=100, subsample=0.6500000000000001)
Total runtime: 63.685036884999136s
Average Accuracy Score: 0.9722222222222222
Best Accuracy Score: 0.9928571428571429
"""

#

import autosklearn.classification
cls = autosklearn.classification.AutoSklearnClassifier()
cls.fit(X_train, y_train)
y_pred = cls.predict(X_test)