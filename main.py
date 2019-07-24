import os, requests, plotly, argparse, sys, wikipedia, json, re
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
    """ Creates a directory for putting the scraped data into"""

    data_directory = os.path.join(ROOT_DIR, "web_scraped_data")
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)


def soupify(website):
    """ Turn html.text into a BeautifulSoup object """

    # Get the html through the Requests module
    html = requests.get(website)
    html.raise_for_status()

    # Turn the Website into a soup
    soup = bs(html.text, "lxml")

    return soup


def extract_recipes_links(website):
    """ Creates a list of all the links given a website"""
    pattern = website[:24] + "featured_item"  # pattern to search href for
    soup = soupify(website)

    # returns <class 'bs4.element.ResultSet'> from the .find_all() method
    # To have to use the .get('href') method to get the specified links
    hrefs = [link.get("href") for link in soup.find_all("a", href=re.compile(pattern))]

    return hrefs


def recipes_to_json_file(website):
    """ Using the list of links, parse the recipe components into a json file 
	and write to disk"""
    hrefs = extract_recipes_links(website)

    scraped_recipe_data = {}

    for recipe_link in hrefs:
        recipe_soup = soupify(recipe_link)
        recipe_name = recipe_soup.h1.string

        # Add into the dictionary
        scraped_recipe_data["recipe"] = recipe_name

        # each recipe has: ingredients, directions, tags
        scraped_recipe_data["recipe"]["ingredients"] = [
            i for i in recipe_soup.find_all("h2", string="Ingredients:").string
        ]
        scraped_recipe_data["recipe"]["instructions"] = [
            i for i in recipe_soup.find_all("h2", string="Directions:").string
        ]
        scraped_recipe_data["recipe"]["tags"] = [
            i for i in recipe_soup.find_all("span").string
        ]
        print(recipe_name)

        # TODO: Store the structured results of the scrape into a dictionary
        # 1. Store the scraped data into a scraped_recipe_data dictionary
        # 2. Have

        # TODO: Write out dictionary to a json file on disk with the appropriate name
        # with open(os.path.join(data_directory, {}+"_recipe.json"), "w").format(str(recipe_name)) as json_file:
        # 	json.dump(scraped_recipe_data, json_file)


def main():

    prepare_data_directory()

    # extract_recipes_links(WEBSITE).recipes_to_json_file()

    recipes_to_json_file(WEBSITE)

    print("Run Complete!!!")


if __name__ == "__main__":
    main()

# find some dataset to practice data wrangling and cleaning on by next week for wednesday
# have the scraper done by Sunday
# see if I can get a simple scraper to get the recipes from the the json feasible
