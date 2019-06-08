import requests, os, re
from bs4 import BeautifulSoup as bs

'''
find_all() Signature: (name, attrs, recursive, string, limit, **kwargs)

'''

html_doc = """
<html><head><title>The Dormouse's story</title></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>

<p class="story">...</p>
"""

def soupify(html):
	soup = bs(html, 'html.parser')
	pretty_soup = soup.prettify()
	return soup, pretty_soup

# you can even define your own functions if the other ones do not work. Pass this in like any parameter into the soup.find_all()
# This function only picks up the <p> pags but doesn't pick up the <a> tags becuase those tags define both "class" and "id". 
# It doesn't pick up tags like <html> and <title> because those tags don't define "class"
def has_class_but_no_id(tag):
	return tag.has_attr("class") and not tag.has_attr("id")

# if you pass in a function to filter on a specific attribute like href, the argument passed into the function will be the attribute value, not the whole tag.
# Here's a function that finds all <a> tags whose href attribute DOES NOT match a regular expression
def not_lacie(href):
	return href and not re.compile("lacie").search(href)

# You can create a function as complicated as you want
# This function returns True if the tag is surrounded by strings
# def surrounded_by_strings(tag):
# 	return (isinstance(tag.next_element, NavigableString) \
# 	 		and ininstance(tag.previous_element, NavigableString))

def main():
	soup, _ = soupify(html_doc)
	last_a_tag  = soup.find("a", id="link3")

	# a generator object is created when you call soup.next_elements. Notice the pluralization of elements. You can iterate over this
	for element in last_a_tag.next_elements:
		print(repr(element)) # returns the canonical string representation of the object

	middle_a_tag = soup.find("a", id="link2")
	for element in middle_a_tag.previous_elements:
		print("#######These are the previous elements of the middle a tag element#######\n{}".format(repr(element)))

	# use .next_element and .previous_element on a bs4.element.Tag to move in between tags
	# print("This is the result the title tag for the soup: {}".format(soup.title)) # returns a navigablestring object
	# print("This is the result of next element for soup: {}".format(soup.p.next_element)) # returns a navigablestring object
	# print("This is the result of next sibling for soup: {}".format(soup.p.next_sibling)) # returns a navigablestring object
	
	# the .find_all() function returns a list of the elements that match the specified parameters
	z = find_all_a_tag = soup.find_all("a") # returns a list of elements

	# you can use the re module to pass in regular expressions to beautifulsoup
	# you can pass in a regular expression object directly to the .find_all() method of the soup
	# this snippet will return a list of all elements that start with the letter "b"
	print("These are the names of the tags that start with the letter b:")
	for tag in soup.find_all(re.compile("^b")):
		print(tag.name)

	# This snippet finds all tags whose names contain the letter t
	print("These are the names of the tages that contain the letter t")
	for tag in soup.find_all(re.compile("t")):
		print(tag.name)

	# This snippet will return a LIST of tags that are contained within the list. Both item a, and item b containees will be returned as a list of elements
	list_to_look = ["html", "a"]
	y = soup.find_all(list_to_look)
	print("This is the list of elements with tags contained in the list_to_look: \n{}".format(y))

	# The value True matches everything that it cn. This code finds all the tags in the document, but none of the text strings contained inside
	print("This is the result of putting True to find_all: \n{}".format(soup.find_all(True)))

	# Using the has_class_but_no_id custom function
	print("This is the result of adding in our own filter function to to the soup: \n{}".format(soup.find_all(has_class_but_no_id)))

	# Using the href attribute value that matches this regular expression evaluation
	print("This is the result of filtering by not_lacie for the href attribute: \n{}".format(soup.find_all(href=not_lacie)))

	# Using the surrounded_by_strings custom function to return a tags that are surroudned by strings
	# print("This is the list of tags that are surrounded by strings: \n{}".format(soup.find_all(surrounded_by_strings)))


if __name__ == "__main__":
	main()

