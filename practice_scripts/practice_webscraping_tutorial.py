'''
This script follows along a tutorial from "automating the boring stuff with python": 
https://automatetheboringstuff.com/chapter11/

It will follow along with the blog, as well as try and implement it in our own way
'''

import requests, bs4, webbrowser, re, os, selenium

# webbrowser.open("http://www.inventwithpython.com/")

'''
Our program will:
1. Get a street address from the command line arguments or clipboard
2. open the web browser to the Google Maps page for the address

This means that your code will need the following:
1. read the command line arguments from sys.argv
2. read teh clipboard contents
3. call the webbrowser.open() function to open the web browser
'''