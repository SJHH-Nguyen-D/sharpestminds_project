import requests, os, re
from bs4 import BeautifulSoup as bs

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

def has_six_characters(css_class):
	return css_class is not None and len(css_class) == 6

def is_the_only_string_with_a_tag(s):
	'''Returns True if this string is the only child of its parent tag'''
	return (s == s.parent.string)

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

	# NAVIGATING the parse tree by using tag names:
	print("This is the prettified navigation print: \n{}".format(soup.body.b.prettify()))
	
	# using a tag name as an attribute will return only first instance of the tag by that name
	print("First element instance with the <a> tag: \n{}".format(soup.a.prettify())) 

	# if you need to get all the elements with the <a> tag, or anything more complicated than the first tag with a certain name,
	# you'll need to use one of the tree search methods:

	# tag's children are available in a list called .contents
	print("This is the list of contents UNDER the <{}> tag: \n{}".format(soup.head.name, repr(soup.head.contents[0])))
	print("This is the list of contents UNDER the <{}> tag: \n{}".format(soup.title.name, repr(soup.title.contents[0])))

	# The BeautifulSoup object itself has children. In this case, 
	# it shouldbe the <html> tag is the child of the beautiful soup object
	print(soup.contents[0].name)

	# The string contents between the <b> tags
	print(repr(soup.find_all("b")[0].contents[0]))
	# .find_all() Signature: soup.find_all(attrs={"key":value"}}

	# returns the a list of elements with the <a> tags
	# If there is a "class" attribute, you can address it as "class_" to avoid naming collisions
	# Using class as a keyword argument will give you a syntax error. As of beautifulsoup, you can search a CSS class by uysing the keyword argument class_
	print(soup.find_all("a", class_="sister"))
	print("The list of elements with the class argument with 'itl in it': \n{}".format(soup.find_all(class_=re.compile("itl"))))

	# using the custom mask function to filter through CSS classes
	print("This is the return of the has six characters mask: \n{}".format(soup.find_all(class_=has_six_characters)))

	# a single tag can have multiple values for its "class" attribute. When you search for a tag
	# that matches a certain CSS class, you're matching against any of its CSS classes
	css_soup = bs('<p class="body strikeout"></p>', "lxml")
	print("This is the result of the CSS example search: \n{}".format(css_soup.find_all("p", class_="strikeout")))
	print("This is the result of the CSS example search: \n{}".format(css_soup.find_all("p", class_="body")))

	# searching for variants of string value doesn't work
	print("Partial matches will return an empty list: \n{}".format(css_soup.find_all("p", class_="strikeout body"))) # returns an empty list

	# you can search for tags that match two or more CSS class,
	# you should use a CSS selector.
	# the select says: inside the <p>, look for a string containing the string "body"
	print(css_soup.select("p.strikeout.body")) # the soup.select() method returns a list of matching elements

	# STRING ARGUMENT
	# With string attribute, you can search for strings instead of tags. As with name, and the keyword arguments, yoiu can pass in a string,k a regular expression, a list or a function or the value True
	# Here are some examles of those:
	print("This is the result of using the string attribute search on the soup: \n{}".format(soup.find_all(string="Elsie"))) # returns a list of the strings which contain the specified string
	listofstringnames = list(["Tillie", "Elsie", "Lacie"])
	print("You can also search for a list of strings contents that are contained in the list provided to the string argument: \n{}".format(soup.find_all(string=listofstringnames)))
	print("Returning a list containing strings that are the only strings with a tag: \n{} ".format(soup.find_all(string=is_the_only_string_with_a_tag)))

	# Although strings only finds strings, you can chain it with other arguments that find tags: Beautfiul soup will find all tags whose .string matches your value for string.
	# This code finds the <a> tags whose .string is "Elsie"
	# String is new to beautifulsoup. It used to be called "text"
	print(soup.find_all("a", string="Elsie"))

	# LIMIT ARGUMENT
	# Find_all() returns all the tags and strings athat match your filters. Thi8s can take a while if the documentis large.
	# If you don'ty need all the results, you can pass in a number for limit. This works
	# Just like the LIMIT keyword in SWWL. It tells Beautiful soup to stop gathering results after it's found certain number
	print("Display the first two elements with the <a> tag in the soup: \n{}".format(soup.find_all("a", limit=2)))

	# You can also find the descendents of a certain tag 
	# use mytag.find_all()
	print(soup.html.find_all("head")[0].prettify())
	# if you only want to find the direct/immediate children of elements within that tag
	# you set the recursive argument to False in .find_all()
	# recursive is set to True by default
	print(soup.find_all("title", recursive=False))

	# Calling a tag is like calling find_all()
	# These two lines are equivalent
	print("This is the result of soup.findall(a): \n{}".format(soup.find_all("a")))
	print("This is the result of soup.a: \n{}".format(soup.a))
	print("This is the result of soup(a): \n{}".format(soup("a"))) # this returns a list of elements containing the "a" tag

	# These two lines are also equivalent
	print("This line is equivalent to the below line: \n{}".format(soup.title.find_all(string=True)))
	print("This line is equivalent to the above line: \n{}".format(soup.title(string=True)))

	# FIND()
	

if __name__ == "__main__":
	main()

