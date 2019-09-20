import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from constants import POSTPROCESSED_DATAPATH


# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv(POSTPROCESSED_DATAPATH, sep=",", header="infer")
features = tpot_data.drop("job_performance", axis=1).values
training_features, testing_features, training_target, testing_target = train_test_split(
    features, tpot_data["job_performance"].values, random_state=42
)

# Average CV score on the training set was:-51844.34650603313
exported_pipeline = make_pipeline(
    SelectFromModel(
        estimator=ExtraTreesRegressor(
            max_features=0.35000000000000003, n_estimators=100
        ),
        threshold=0.15000000000000002,
    ),
    PCA(iterated_power=2, svd_solver="randomized"),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    MaxAbsScaler(),
    RandomForestRegressor(
        bootstrap=False,
        max_features=0.7000000000000001,
        min_samples_leaf=2,
        min_samples_split=7,
        n_estimators=100,
    ),
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
scores = exported_pipeline.score(testing_features, testing_target)

predictions_outfile = pd.DataFrame()
predictions_outfile["Truth Label"] = testing_target
predictions_outfile["Prediction"] = results
predictions_outfile["Coefficient of Determination R^2"] = scores
predictions_outfile.to_csv("./prediction_results/tpot_prediction_results.csv")
