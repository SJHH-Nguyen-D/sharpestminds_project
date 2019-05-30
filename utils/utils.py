import os
import pandas as pd 
import numpy as np 

data_loader(datafile):
	data = pd.read_csv(datafile, header=None, names=names)
	print("Data Loaded")
	return data

