import requests, os
from bs4 import BeautifulSoup

'''
Once your program has downloaded a webpage using the requests module, you will have the page's HJTML content as 
a single string value. Now you need to fgure out which part of the HTML corresponds to the information on the web page you're interested in

This is where the browser's developer tools can help. Say you want to write a program to pull weather forecast data from https://weather.gov/. Before
writing any code, do a little research. If you visit the site and search for the 9415 ZIP code, the site will take you to a page showing the forecast for that area.

What if you're interested in scraping the temperature information for that ZIP code? Righ-click where it is on the page and 
select Inspect Element from the context menu that appears. This will bring up the Developer Tools window, which shows you the HTML that produces this particular part of the web page.
'''

'''
Parsing HTML with Beautiful Soup

Beautiful soup is a module fro extracting HTML from an HTML page ( and is much better for this purpose than regular expressions). 
While beautiful soup is the name of the package, to import it you use import bs4.

The bs4.BeautifulSoup() fuinction needs to be called with a string containing the HTML it will parse.
The bs4.BeautifulSoup() function returns a BeautifulSoup object.
Enter the following into the interactive shell while your computer is connected to the internet
'''

res = requests.get("http://nostarch.com")
res.raise_for_status()

nostarchsoup = BeautifulSoup(res.text)

print(type(nostarchsoup))

'''
This code uses requests.get() to downlaod the main page frtom the No Starch Press website and then
passes the text attribute of the response to the BeautifulSoup().
The BeautifulSoup object that it returns is stored in a variable named nostarchsoup.

You can load an HTML file from your harddrive by passing a File objecft to
bs4.BeautifulSoup(). Enter the following into the interactive shell.
'''

examplefile = open("example.html")
examplesoup = BeautifulSoup(examplefile)
print(type(examplesoup))

'''
Finding an element with the select() method

Use the select() element from the beautifulsoup object to select
a component of the text that fits a particular pattern, much like using
regular expression patterns.

Selector passed to the select() method

Will match...

soup.select('div')

All elements named <div>

soup.select('#author')

The element with an id attribute of author

soup.select('.notice')

All elements that use a CSS class attribute named notice

soup.select('div span')

All elements named <span> that are within an element named <div>

soup.select('div > span')

All elements named <span> that are directly within an element named <div>, with no other element in between

soup.select('input[name]')

All elements named <input> that have a name attribute with any value

soup.select('input[type="button"]')

All elements named <input> that have an attribute named type with value button
'''

examplefile = open("example.html")
examplesoup = BeautifulSoup(examplefile.read())
elems = examplesoup.select("#author")
print(type(elems))
print("This is the number of elements in the elems list: {}".format(len(elems)))
print(elems[0])

print("This is the text contained within the matched element: {}".format(elems[0].getText()))

print(str(elems[0]))
print("These are the attributes of the matched element: {}".format(elems[0].attrs))

'''
You can also pull the <p> elemnts from the soup object. Enter this into the interactive shell
'''

pelems = examplesoup.select("p")
print("This is how many elements are in pelems: {}".format(len(pelems)))
print(str(pelems[0]))
print(str(pelems[0].getText()))
print(str(pelems[2].getText()))

'''
Getting data from an element's attributes

The get() method for Tag objects makes it simple to access attribute values from an element. The method is passed 
a string of an attribute name and returns that attributes value.
Using example.html, enter the following into the interactice shell
'''

soup = BeautifulSoup(open("example.html"))
spanelem = soup.select("span")[0]
print(str(spanelem))
print("There are {} span elements in this html file".format(len(soup.select("span"))))

print(spanelem.get("id")) #the value for the key/attribute "id" is "author"

print("some_nonexistent_addr is NOT in the spanelem list: {}".format(spanelem.get("some_nonexistent_addr") == None))

# see the attributs of the span elems list
print(spanelem.attrs)

