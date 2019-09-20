# load in new postprocessed dataset
import pandas as pd
import timeit
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
from constants import PERIODIC_CHECKPOINT_FOLDER, POSTPROCESSED_DATAPATH
import os


df_sfs = pd.read_csv(POSTPROCESSED_DATAPATH)

# Train/test split using the new 10 feature selected dataset
X_train, X_test, y_train, y_test = train_test_split(
    df_sfs.loc[:, df_sfs.columns != "job_performance"].values,
    df_sfs["job_performance"].values.ravel(),
    test_size=0.25,
    random_state=42,
)

y_train = y_train.ravel()
y_test = y_test.ravel()

## TPOT Model Performance

tpot_regressor_pipeline_selector = TPOTRegressor(
    generations=20,
    population_size=50,
    offspring_size=None,
    cv=10,
    random_state=42,
    verbosity=2,
    memory="auto",
    warm_start=True,
    use_dask=False,
    periodic_checkpoint_folder=PERIODIC_CHECKPOINT_FOLDER,
)

tpot_regressor_pipeline_selector.fit(X_train, y_train)
y_pred = tpot_regressor_pipeline_selector.predict(X_test)


def save_best_pipeline(selected_pipeline, filename):
    selected_pipeline.export(os.path.join(PERIODIC_CHECKPOINT_FOLDER, f"{filename}.py"))


# tpot_regressor_pipeline_selector.export(os.path.join(PERIODIC_CHECKPOINT_FOLDER,'tpot_exported_pipeline.py'))

save_best_pipeline(tpot_regressor_pipeline_selector, "tpot_exported_pipeline")
