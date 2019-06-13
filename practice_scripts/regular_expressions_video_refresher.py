# Regular Expression Video Tutorial
import re, json, requests, numpy as np, webbrowser

sentence = "Start a sentence and bring it to an end."

# a raw string is just a string prefixed with an r and that tells python not to handle backslashes in any special way
# for this tutorial, and for working with regular expressions in general, you will be working with raw strings a lot.
print(r'\tTab')
print('\tTab')

# RE.COMPILE()
# the re.compile() method allows us to separate out our pattern into a variable to make it easier to reuse that variable to perform multiple searches
# Finds this exact pattern in this exact sequence
pattern = re.compile(r"abc")

# Now that we have that pattern specified, we want to search through our literal text with that pattern.
# Now we want to create a variable, traditionally called matches, which will return an object that contains all the matches when searching against the 
# specified pattern

text_to_search = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRISTUVWXYZ'

# the finditer() method gathers all the matches in an easy to read format
matches = pattern.finditer(text_to_search) # the .finditer() method returns a callable_iterator object

# The span is the beginning and end index of the match
# It only found 1 match of 'abc' and it found it in our alphabet 10 through 13, with 0-based indexing
# Finds the exact pattern in that exact sequence
for i in matches: # each i is a sre.SRE_MATCH object with attributes: {span=(), match="pattern"}
	print(i)
	print(i[0])
# print(text_to_search[10:13]) # just to check if our span of our text_to_search matches up with the results of our .finditer() function

email = r"dennisnguyendo@gmail.com"

pattern = re.compile(r"dennisnguyendo@gmail\.com")

matches = pattern.finditer(email)
for i in matches:
	print(i[0])

'''
. - matches any character except a new line
\d - Digit (0-9)
\D - Not a Digit (0-9)
\w - word character (a-z, A-Z, 0-9, _)
\W - not a word character
\s - whitespace (space, tab, newline)
\S - not a whitespace character(space, tab, newline)
\b - word boundary
\B - not a word boundary

'''

pattern = re.compile(r"\d") # compare the different regular expression unique search patterns you can searchb for in a string
matches = pattern.finditer(text_to_search) 
listofnumbers = [i[0] for i in matches] # list of the digits that appear in the text_to_search string
print(listofnumbers)

pattern = re.compile(r"\D") # compare the different regular expression unique search patterns you can searchb for in a string
matches = pattern.finditer(text_to_search) 
listofnonnumberchars = [i[0] for i in matches] # list of the letter characters that appear in a string
print(listofnonnumberchars)

pattern = re.compile(r"\w") # compare the different regular expression unique search patterns you can searchb for in a string
matches = pattern.finditer(email) 
listofwordcharacters = [i[0] for i in matches] # list of the a-z, A-Z, 0-9, and _ characters that appear in a string
print(listofwordcharacters)

pattern = re.compile(r"\W") # compare the different regular expression unique search patterns you can search for in a string
matches = pattern.finditer(email) 
listofnotawordcharacters = [i[0] for i in matches] # list of the NON - a-z, A-Z, 0-9, and _ characters that appear in a string
print(listofnotawordcharacters)

# Let's say we want to search up a string with a word boundary in a particular string
particular_string = "Ha, HaHa"
pattern = re.compile(r"\bHa")
matches = pattern.finditer(particular_string)
for i in matches:
	print(i)
# notice here that the finditer() method is matching only the first instance of the word and the second instance because the third instance
# of Ha, doesn't have a word boundary

# We are now using the special characters to indicate where we want to search for in the string
sentence = "Start a sentence and bring it to an end."
pattern = re.compile(r"^Start") # this is to say that we want to search from the beginning of the string and we specify that we are looking at the literal start
								# of the string, with the literal raw character pattern specified
matches = pattern.finditer(sentence)
for i in matches:
	print(i)

# Now, if we use the ^ symbol again with a different search pattern, we get something empty because the string doesn't literally start with that pattern
sentence = "Start a sentence and bring it to an end."
pattern = re.compile(r"^a") 
matches = pattern.finditer(sentence)
for i in matches:
	print("This is the result of search for the pattern ^a: {}".format(i)) # None is returned, therefore nothing is returned

# To search for a patttern at the end of a string, we use the $ sign after our specified pattern
sentence = "Start a sentence and bring it to an end."
pattern = re.compile(r"end.$") 
matches = pattern.finditer(sentence)
for i in matches:
	print(i) # however, there are no matches and objects returned as a result of an inexact match

