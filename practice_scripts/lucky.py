'''
Project: "I'm feeling lucky" Google Search

Whenever I search a topic in Google, I don't look at just one search result at a time. By middle-clicking a search result, I hope the link into a new tab.
I search google often enough that this worflow, opening my browser, searching for a topic, 
and middle-clicking several links one by one- is tedious. It would be nice
if I coiuld simply type a search term in the CLI and have my computer
automatically open a browser with all the top search results in a new tabs.
Let's write a script to do this.

This is what your program does:
1. Gets search keywords from the command line arguments
2. Retrieves the search results page
3. Opens a browser tab for each result

This means that your code will need to do the following:
1. Read the command line arugments from sys.argv
2. Fetch the search result page with the requests module
3. Find the links to each search result
4. Call the webbrowser.open() function to open the web browser
'''


'''
Step 1: Get the Command line arguments and request the search page

Before coding anything, you first need to know the URL fo the search result page.
By looking at the browser's address bar after doing a Google search, you can see that the result page has a URL like
https://www.google.com/search?q=SEARCH_TERM_HERE. The requests module can download this page
and then you can user Beautiful SOup to find the search result links in the HTML.
Finaslly, you'll use the webbrowser module to open those links to browser tabs

Executing in the command line would look like this:

#! python3
lucky.py

'''

import requests, os, sys, argparse, pyperclip, webbrowser
from bs4 import BeautifulSoup as bs 

print("Googling...")
res = requests.get("https://www.google.com/search?=" + " ".join(sys.argv[1:]))
if res.raise_for_status() == None:
	print("Code 200: Succeeded without Errors")
else:
	print("Code 404: Failed")


# TODO: Retrieve Top search result links
# TODO: Open a browser tab for each result

'''
# Step 2: Find all the results

Now, you need to use Beautiful soup to extract the top search result links from your downloaded HTML. But how do you figure out the right selector for the job:
For example, you can't just search for all the <a> tags, because there are a lot fo links
you don't care about in the HTML. Instead, you just inspect the search result page, with the browser's
developer tools to try and find a seleector that will pick out only
links you want.

After doing a google search for BeautifulSoup, you cvan open the browser's developer tool sand inspect some of the elements on the page.
They look incredibly complicated, something like this: <a href="/url?sa..."</a>

It doesn't matter that the element looks incredibly complicated. You just need to find the pattern that all the search result links will have. But this <a> element doesn't
have anything that easily distinguishses it from the nonsearch result <a> elements on the page.
'''

# https://automatetheboringstuff.com/chapter11/

# Retrieve top search result links.
soup = bs(res.text)
print(soup)

# open a browser tab for each element
linkelems = soup.select(".r a")

'''
The CSS class "r" is only used for search result links. We are going to use it as a marker for the <a> element you are lookng fro.
You can create a BeautifulSoup objet from the downloaded page's HTML text and then use the selector ".r a" to find all <a> elements that are within an element that has
the CSS class "r"
'''

'''
Step 3: Open Web Browsers for each result
'''

numopen = min(5, len(linkelems))
for i in range(numopen):
	print("Opening: \n{} ".format("https://wwww.google.com" + linkelems[i].get("href")))
	webbrowser.open("https://wwww.google.com" + linkelems[i].get("href"))

'''
By default, you open the first five search results in new tabs using the webbrowser module. However, the user may have searched for something that turned up fewer than 5 results. The soup.select()
call returns a list of all the elements that matched your ".r a" selector, so the number of tabs you want to open is either 5 or the length of this list(whichever is smaller)

The built-n Python function min() returns the smallest of the integer or float arguments it is passed. (There is also a built-in max() function that returns the largest 
argument that is passed). You can use min() to find out whether there are fewer than five links in the list and store the number of links to open in a variable named numopen.
Then you can run through a for loop by calling range(numopen)

For each iteration of the loop, you use webbrowser.open() to open a new tab in the web browser. Note that the href attribute's value in the returned <a> elements do not have the intial http://google.com part, 
so you have to concatenate that to the href attributes string value.

Now you can instantly open the firstfive google results for, say, python programming tutorial by running:

$ lucky.py programming tutorials 


You can extend this idea to other applications. The benefit of tabbed browsing is that you can esily open links in new tabs to peruse later.
A program that automatically opens several links at once can be a nice shortcut to do the following: 
1. open all the product pages after search a shopping site such as amazon
2. open all the links to reviews for a single product
3. Open the result links to photos after performing a search on a photo site such as Flickr on Imgur
'''

