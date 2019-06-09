import pandas as pd 
import os, requests, plotly, argparse, sys, wikipedia, json
from bs4 import BeautifulSoup as bs
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

def prepare_data_directory():
	data_directory = os.path.join(ROOT_DIR, "web_scraped_data")
	if not os.path.exists(data_directory):
		os.mkdir(data_directory)

def soupify(website):
	# Get the html through the Requests module
	html = requests.get(website)
	html.raise_for_status()

	# Turn the Website into a soup
	soup = bs(html.text, 'lxml')
	return soup

def extract_recipes(website):
	soup = soupify(website)

	list_of_hrefs = []

	for link in soup.find_all("a"):
		list_of_hrefs.append(link.get('href'))

	pattern = "https://entomofarms.com/featured_item"
	recipe_links = [i if pattern in i for i in list_of_hrefs[i]]

	recipe_links_list = soup.select('a[href^="https://entomofarms.com/featured_item"]')
	print("This is list of links: \n{}".format(recipe_links_list))
	print("This is the list: \n{}".format(recipe_links))
	return recipe_links_list

# TODO: To JSON files
# def write_scraped_to_json(dataframe):
# 	# write to json instead of pandas.... back to the drawing board
# 	with json.open(os.path.join(data_directory, ".json"), sep=',', encoding='utf-8') as jsonwriter:
# 		json = ""
# 	return json

def main():
	prepare_data_directory()
	extract_recipes(WEBSITE)
	print("Run Complete!!!")

if __name__ == "__main__":
	main()

# find some dataset to practice data wrangling and cleaning on by next week for wednesday
# have the scraper done by Sunday
# see if I can get a simple scraper to get the recipes from the the json feasible
