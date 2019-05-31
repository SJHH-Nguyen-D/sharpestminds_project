# Downloading the Files from the WEb with the Requests Module

import requests

'''
The requests.get() function takes a string of a URL to download. By calling type()
on requests.get()'s return value, you can see that it returns a Response object, 
which tonains the response that the web server gave for your request. I'll explain
the Response object in more detail later, but for now, enter the following into the interactive shell while
your computer is connected to the Internett:
'''

res = requests.get("https://www.automatetheboringstuff.com/files/rj.txt")
print(type(res))

print("It is {} that the request status is successful".format(res.status_code == requests.codes.ok))
print("The text is {} characters long".format(len(res.text)))
print("These are the first 350 characters from the text: \n{}".format(res.text[:350]))

'''
Checking for Errors

As you've, the Reponse object has a status_code attribute that can be checked against request codes.ok to see whether the download succeeded. A simpler way to check for success is to call the raise_for_status() method on the Response object. 
This will raise an exception if there was an error downloading the file and will do nothing if the download succeeded. Enter the following in the interactive shell.

The raise_for_status() method is a good way to ensure that a program halts if a bad download occurs. This is a good thing: You want your program to stop as soon
as some unexpected error happens. If a failed download isn't a deal breaker for your program, you can wrap the raise_for_status() line with
try and except statements to handle this error case without crashing
'''

res = requests.get("https://inventwithpython.com/page_that_does_not_exist")
try:
	res.raise_for_status()
except Exception as exc:
	print("There was a problem: \n{}".format(exc))

'''
Saving the downloaded files to the Hard Drive

From here, you can save the web page to a file on your hard drive with the standard open() function and write() method,
There are some slight difference, though. First you must open the file in write binary mode by passing the string "wb" asd the second argument to open().
Even if the page is in plaintext (such as the Romeo and Juliet text you just downloaded earlier), you need to write binary data 
instead of text data in order to maintain the Unicoding encoding of the text.

To write the web page to a file, you can use a for loop with the response object's iter_content() method.
'''

import requests
res = requests.get("https://automatetheboringstuff.com/files/rj.txt")
res.raise_for_status()

import os

if not os.path.exists(os.path.join(os.getcwd(), "webdata")):
	os.mkdir(os.path.join(os.getcwd(), "webdata"))

if not os.path.isfile(os.path.join(os.getcwd(), "webdata", "RomeoAndJuliet.txt")):
	read_size = 100000
	playFile = open("./webdata/RomeoAndJuliet.txt", "wb")
	for chunk in res.iter_content(read_size):
		playFile.write(chunk)
	playFile.close()

'''
The iter_content() returns chunks of the content on each iteration through the loop.
Each chunk is of the bytes datatype, and you get to specify how many bytes each chunk will contain.
One hundred thousand bytes is generally a good size, so 100k byes as the argument to res.iter_content()

The file RomeoAndJuliet.txt will not exist in the current working directory. Note that while the filename on the website
was rj.txt, the file on your hard drive has a different filename. The requests module simply handles downloading the contents of web pages. Once the page is downloadd, 
it simply downloads data in your program. Even if you were to lose your internet connection after downloading the web page, all the page data
would still be on your computer.

The write() method returns the number of bytes written to the file. In the previous
example., there was 100k bytes in the first chunk, and the remaining part of the file needed only 78,981 bytes.

To review, here's the complete process for downloading and saving a file:

1. call requests.get() to download the file
2. Call open() with "wb" to create a new file in write binary mode
3. Loop over the Response object's iter_content() method.
4. CAll write() on each iteration to write the content to the file
5. call close() to close the file.

'''