import os, json, webbrowser, selenium
import pandas as pd 
import numpy as np 
import wikipedia as w

data_loader(datafile):
	data = pd.read_csv(datafile, header=None, names=names)
	print("Data Loaded")
	return data

# Wikipedia Web Scraper
def wiki_webscraper(list_of_search_terms):
	
	dataframe = pd.DataFrame(columns=['key_search_term', 'summary'])
	# start off with getting the summary of the texts of the 
	for search_term in list_of_search_urls:
		term_summary = w.summary(search_term, sentences=10)
		# res = requests.get(list_of_search_urls[search_link])
		# res.raise_for_status()
		# soup = BeautifulSoup(res.text)
		df.loc[list_of_search_urls.index(search_term)] = [search_term, term_summary]
		return json