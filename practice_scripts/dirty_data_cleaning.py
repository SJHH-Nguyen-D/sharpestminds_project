# Data Preprocessing and Cleaning Script

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os, requests, webbrowser
import xlrd
from datetime import datetime

datafile = os.path.join(os.path.dirname(os.getcwd()), "dataset", "air-quality-london-mean-roadside.xlsx")
sheet_name = "london-mean-roadside"

# df.keys() = Index(['Month (text)', 'GMT', 'Nitric Oxide (ug/m3)',
#        'Nitrogen Dioxide (ug/m3)', 'Oxides of Nitrogen (ug/m3)',
#        'Ozone (ug/m3)', 'PM10 Particulate (ug/m3)',
#        'PM2.5 Particulate (ug/m3)', 'Sulphur Dioxide (ug/m3)'],
#       dtype='object')

# Read in the data
df = pd.read_excel(datafile, sheetname=sheet_name)


# Dealing with missing values for numeric features
from sklearn.preprocessing import Imputer
imp=Imputer(missing_values="NaN", strategy="median" )
df["Nitric Oxide (ug/m3)"]=imp.fit_transform(df[["Nitric Oxide (ug/m3)"]]).ravel()
df["Oxides of Nitrogen (ug/m3)"]=imp.fit_transform(df[["Oxides of Nitrogen (ug/m3)"]]).ravel()

############################### Date Time Columns #########################
import time
# combine the month text with the time text so that we can do the hourly plotting
# first two cols are datetime but separate, each is series of strings

# print(df.iloc[:,0])
# print(df.iloc[:,:2])

df.apply(lambda r : pd.datetime.combine(r['Month (text)'] ,r['GMT']), 1)

print(df.head())