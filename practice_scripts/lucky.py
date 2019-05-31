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

