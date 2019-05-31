'''
This script launches a map in the browser using an address from the 
command line or clipboard and does a google search with the address provided at the commandline
'''

import bs4, requests, selenium, sys, os, webbrowser


'''
Command line arguments are usually separated by spaces, but in this case, 
you want to interpret all of the arguments as a string. Since sys.argv is a list of strings, 
you want to pass it to the join() method, which returns a single string value.
You don't want the program name in this string, so instead of sys.argv, you should
ass sys.argv[1:] to chop off the first element of the array. The final string that this expression 
evaluates to is stored in the address variable
'''

if len(sys.argv) > 1:
	# Get the address from the command line
	# The sys.argv variable stores a list of the program's filename and command line arguments.
	# this check is to see if command line arguments are provided
	address = " ".join(sys.argv[1:])


# step 3: handle the clipboard content and launch the browser
# this module allows you to pull from the clipboard
import pyperclip

if len(sys.argv) > 1:
	address = " ".join(sys.argv[1:])
else:
	# get the address from the clipboard
	address = pyperclip.paste()

webbrowser.open("https://www.google.com/maps/place/" + address)

'''
If there are no command line arguments, the program will assume the address is stored
on the clipboard. You can get clipboard content with pyperclip.paste() and store it in a variable named address.
Finally, to launch a web browser with the google maps URL, call webbrowser.open()
'''