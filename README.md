# Sharpest Minds Project

## Background

As part of the Sharpest Minds mentorship program, mentees choose to develop an end-to-end data science project on their own with the guidance of an experienced data scientist mentor from the organization. An end-to-end project comprises of acquiring your own dataset by a variety of methods; preprocessing the data; exploring the data; feature selection and engineering; model selection, model training, model validation; and parameter tuning. The ultimate goal of the project would be to provide inferences for a user through a web API or application, using the Flask web framework API, given user inputs on employee characteristics.


![Employees, yay!](images/stock_image.jpg "Employees are having a good time.")

## The Problem

The data set in question is 92 Mb in size with a shape of (20000, 380). Each sample is an employee record with attributes on their education, demographics, type of employment. The target attribute is a computed relative employee performance score. The objective then is to produce inferences of a employee performance metric, based on the characteristics of each employee.

## Requirements

The module requirements for the project can be installed using ```pip install -r requirements.txt ```.

## The Project Build

The project is constructed end-to-end with the following steps:

* data acquisition
* data exploration and visualization
* data preprocessing
    * Dropping
    * Encoding
    * Imputing missing values
* feature selection
* modeling
* pipeline optimization and selection
* deployment as web API

## Data Acquisition

The dataset was acquired as part of a dataset provided by my Sharpest Minds mentor.

## Data Exploration and Visualization

The data exploration and visualization can be seen in the accompanying Jupyter Notebook ```data_visualizations.ipynb```. 

## Data Preprocessing

The raw employee performance dataset is stored locally, loaded in as a Pandas DataFrame and preprocessed through a series of encoding, dropping and selection steps, performed heuristically, which is detailed in the ```preprocessing.py```. The script makes use of pandas.Categorical() method, scikit-learn.LabelEncoder() and OneHotEncoder() before the features selection step. Additionally, instead of a uniform, static method for computing for missing values, an machine-learning approach to imputation is performed with ```missingpy```'s nearest neighbors imputer.

## Feature Selection

The feature selection for the model was performed using a combination of knowledge-based, hand-selected and the ```SequentialFeatureSelector``` in the ```mlxtend``` module and is part of the ```preprocessing.py``` script. The number of features that were selected, prior to modeling, was arbitrary. 

![Sequential feature selector](images/stairs_resized.jpeg "Selecting one feature at a time.")

## Pipeline Optimization and Selection

The final model used to produce inferences used the [TPOT algorithm](https://epistasislab.github.io/tpot/), which makes use of genetic programming to select the best model and hyperparameters for this regression problem. Running this portion of the project took the longest, as 1050 pipelines were trialed to output the final selected pipeline. This pipeline is available in ```tpot_exported_pipeline.py```, and used to produce inferences for a user, given input, in the Flask web app.

## Deployment

A user can receive inferences for a job performance score via the Flask application on the [website](sjhh-nguyen-d.github.io) by inputting their own employee charactersistics in the input fields, using the trained model in the ```tpot_exported_pipeline.py```. The scores are an arbitrary, relative the employee performance scores in the dataset. Users are also able to see their score relative to other data points on the distribution, grouped by several categorical features. Please do not be offended by your given score, as they are fictional and have no baring on your reality.
