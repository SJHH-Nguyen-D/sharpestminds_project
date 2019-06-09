import pandas as pd 
import os, requests, plotly, argparse, sys, wikipedia, json
from bs4 import BeautifulSoup
import wikipedia as w

from utils.constants import *

# TODO: Arg Parser
# TODO: save the html so taht you don't have to hit the source site again  (this is a big burden if you are sending out thousands of requests per second)
# TODO: the above step is not feasible if there are updates to the page, so that your scraper breaks.
# TODO: work off the the html
# Read beautiful soup documentation

# TODO: shape, Prep and Save Scraped data onto the hard disk
# TODO: Data Loader

# Read through the Beautiful Soup documentation that was linked to understand how the scraping works
# Go and find more structured websites you can go and work with

# TODO: Data Preprocessing/Cleaning
# TODO: Data Visualization
# TODO: EDA
# TODO: Modelling

# TO THINK ABOUT: break tasks into subtasks

'''
This main script will be used to put everything together. 

I eventually want to put the data acquisition and prep scripts into the utils folder
'''

def set_language():
	w.set_lang(language)

def prepare_data_directory():
	data_directory = os.path.join(ROOT_DIR, "wikidata")
	if not os.path.exists(data_directory):
		os.mkdir(data_directory)

def soupify(html):
	soup = bs(html, 'html.parser')
	pretty_soup = soup.prettify()
	return soup, pretty_soup

def extract_recipes(soup, tag):
	soup.
	return elements

def write_scraped_to_json(dataframe):
	set_language()
	# write to json instead of pandas.... back to the drawing board
	with json.open(os.path.join(data_directory, ".json"), sep=',', encoding='utf-8') as jsonwriter:
		json = 
	return json

# class WebScraper():
# 	def __init__(self, url, dataset_directory):
# 		self.url = url
# 		self.dataset_directory = dataset_directory

# 	def scrape_to_json():


def main():
	soup, _  = soupify(WEBSITE)
	prepare_data_directory()
	wiki_webscraper(WIKI_SEARCH_TERMS)
	print("Run Complete")


'''
This code uses requests.get() to download the main page from a site.

Use this to pass to beautifulsoup () to soupify it.

Use then can perform a variety of operations on this soup object including:
	 - select
	 - prettify
'''

if __name__ == "__main__":
	main()

links in the skype


# find some dataset to practice data wrangling and cleaning on by next week for wednesday
# have the scraper done by Sunday
# see if I can get a simple scraper to get the recipes from the the json feasible
