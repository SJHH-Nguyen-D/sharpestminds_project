'''
This script generates a word cloud using NLP for the purpose of EDA 
'''

import pillow, os, wordcloud
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# %matplotlib inline 
import seaborn as sns

# TODO: pairs plot
# TODO: parallel coordinates plot
# If 
# git clone https://github.com/amueller/word_cloud.git
# cd word_cloud
# pip install .

'''
Word cloud will be used t read in  image as the mask for the world cloud
The latest version with the ability to mask the cloud into any shape of your choice requires a different method of installation

The tutorial uses the wine review dataset from Kaggle. This collection is a great dataset for learning with no missing values (which will take time to handle) and a lot fo text \
(wine reviews), categorical and numerical data
'''

from wordlocud import WordCloud, STOPWORDS, ImageColorGenerator
from pillow import Image 
import warnings
warnings.filterwarnings("ignore")

# load in the dataframe
# we used index_col=0 to say that we don't read in row name (index) as a separate column
df = pd.read_csv("./data/winemag-data-130k-v2.csv", index_col=0)

print(df.head())
print("There are {} observations and {} features in this dataset. \n".format(df.shape[0], df.shape[1]))
print("There are {} types of wine in this dataset such as {} ... \n".format(len(df.variety.unique()), ", ".join(df.country.unique()[:5])))
print("There are {} countries producing wine in this dataset such as {} ...\n".format(len(df.country.unique()), ", ".join(df.country.unique()[:5])))

sub_df = df[["country", "description", "points"]]
print(sub_df.head())

'''
To make comparisons between groups of a feature, you can use groupby() and compute summary statistics.
With the wine dataset, you can group by country and look either the summary statistics for all countries' points an dprice of select the most popular and expensive ones.
'''

# groupby by country
country = df.groupby("country") # a dataframe is what is returned

# summary statistic of all countries
print(country.describe().head())

# this selects the top 5 highest average points among all 44 countries
print(country.mean().sort_values(by="points", ascending=False).head())

# plot the number of wines by country using the plot method of pandas dataframe and matplotlib.
plt.figure=(figsize=(15, 10))
country.size().sort_values(ascending=False).plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Number of Wines")
plt.show()

'''
Among 44 countries producing wine, US has more than 50,000 types of wine in the wine review dataset, twice as much as the next one in the rank: France
the country famous for its wine. Italy also produces a lot of quality wine, having nearly 20,000 wines open to review.

Does quantity trump quality?

Let's now take a look at the plot of all 44 countries by its highest rated wine, using the same plotting technique as above:
'''
plt.figure(figsize=(15, 10))
country.max().sort_values(by="points", ascending=False)["points"].plot.bar()
plt.xticks(rotation=50)
plt.xlabel("Country of Origin")
plt.ylabel("Highest point of Wines")
plt.show()

'''
Austrailia, US, Portugal, Italy, and France all have 100 points wine. If you nhotice, Portugal ranks 5th and Australia ranks 9th in the number
pf wines produces in the dataset, and obth countries have less than 8000 types of wine.

That's a little bit of data exploration to get to know the dataset that you are using today. Now you will start to dive into the main course of the mean: WordCloud
'''