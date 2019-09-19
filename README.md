# Sharpest Minds Project

## Background

As part of the Sharpest Minds mentorship program, mentees choose to develop an end-to-end data science project on their own with the guidance of an experienced data scientist mentor from the organization. An end-to-end project comprises of acquiring your own dataset by a variety of methods; preprocessing the data; exploring the data; feature selection and engineering; model selection, model training, model validation; and parameter tuning. The ultimate goal of the project would be to provide inferences for a user through a web API or application, using the Flask web framework API, given user inputs on employee characteristics.

## Data Acquisition

The dataset was acquired as part of a dataset provided by my Sharpest Minds mentor.

## The Problem

The data set in question is 92 Mb in size with a shape of (20000, 380). Each sample is an employee record with attributes on their education, demographics, type of employment. The target attribute is a computed relative employee performance score. The objective then is to produce inferences of a employee performance metric, based on the characteristics of each employee.

## Requirements

The requirements for the project can be installed using ```pip install -r requirements.txt ```.

## Building the Project

The data exploration and visualization can be seen in the accompanying Jupyter Notebook: ```data_visualizations.ipynb``` . 

T

The modelling and pipe

## The Application

A user can receive inferences for a job performance score via the Flask application on the website by inputting employee charactersistics in the input fields. 
