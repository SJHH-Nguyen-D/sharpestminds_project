from bs4 import BeautifulSoup as bs
import os, requests, webbrowser, re


def soupify(website):
	# Get the html through the Requests module
	html = requests.get(website)
	html.raise_for_status()

	# Turn the Website into a soup
	soup = bs(html.text, 'lxml')

	return soup

# def extract_recipes_links(website):
	
# 	pattern = website[:24] + "featured_item" # pattern to search href for
# 	soup = soupify(website)

# 	# returns <class 'bs4.element.ResultSet'> from the .find_all() method
# 	# To have to use the .get('href') method to get the specified links
# 	hrefs = [ link.get('href') for link in soup.find_all("a", href=re.compile(pattern)) ]
	
# 	return hrefs

# def recipes_to_json_file(website):
# 	hrefs = extract_recipes_links(website)

# 	scraped_recipe_data = {}

# 	for recipe_link in hrefs:
# 		recipe_soup = soupify(recipe_link)
# 		recipe_name = recipe_soup.h1.string

# 		# Add into the dictionary
# 		scraped_recipe_data["recipe"] = recipe_name

# 		# each recipe has: ingredients, directions, tags
# 		scraped_recipe_data["recipe"]["ingredients"] = [ i for i in recipe_soup.find_all("h2", string="Ingredients:").string ]
# 		scraped_recipe_data["recipe"]["instructions"] = [ i for i in recipe_soup.find_all("h2", string="Directions:").string ]
# 		scraped_recipe_data["recipe"]["tags"] = [ i for i in recipe_soup.find_all("span").string ]
# 		print(recipe_name)

# 		# TODO: Store the structured results of the scrape into a dictionary
# 		# 1. Store the scraped data into a scraped_recipe_data dictionary
# 		# 2. Have 

# 		# TODO: Write out dictionary to a json file on disk with the appropriate name 
# 		# with open(os.path.join(data_directory, {}+"_recipe.json"), "w").format(str(recipe_name)) as json_file:
# 		# 	json.dump(scraped_recipe_data, json_file)



def main():

	website = 'https://entomofarms.com/featured_item/super-green-cricket-flour-kale-smoothie/'
	soup = soupify(website)

	for i in soup.find_all("h2"):
		if i.contents: # checks to see if the headers are not empty
			a = str(i.contents[0])
			
	# if soup.h2:
	# 	print(soup.h2.string)

	a = soup.find_all("div", class_=re.compile("col-md-6"))
	# print(type(a)) # this is a bveautiful soup.element.resultset object
	# for i in a:
	# 	print(type(i))
	# 	print(i)


	# for i in a:
	# 	print("This is the iterable result of i: \n{}".format(i))

	ingredients = a[0] # bs4.element.Tag
	directions = a[1]

	# # print(a[0])
	# print(type(a[0]))

	# # print(ingredients.contents) # list of elements

	# for i in ingredients.contents:
	# 	# i is elemental tags
	# 	print(i)

	# print(len(ingredients.contents)) # 9
	# print(type(ingredients.contents[7])) # bs4 elements tags
	
	print(ingredients.contents[7].contents)
	
	for i in ingredients.contents:
		print(type(i))
		print(i)
	
if __name__ == "__main__":
	main()

