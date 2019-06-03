'''
XKCD comic image saver:
1. Loads the XKCD home page
2. Saves the comic image on that page
3. Follows the previous comic link
4. Repeats until it reaches the first comic

This means your code will do the following:
1. Download the pages with the requests module
2. Find the URL of of the comic image for a page using Beautiful Soup
3. Download and save the comic image to the hard drive with iter_content()
4. Find the URL fo the "previous" comic link, and repeat
'''

import requests, os, sys, argparse, webbrowser
from bs4 import BeautifulSoup as bs 

'''
Step 1: Design the Program

If you open the brower's developer tools and inspect the elements on the page, you'll find the following
* the URL of the comic's image file is given by the href attribute of an <img>  element
* The <img> element is inside a <div id="comic"> element
* The previous button has a rel HTML attribute with the value prev.
* The first comic's Prev button links to the http://xkcd.com/# URL, indicating that there are no more preivous pages
'''

'''
Step 2: Download the web page
'''

'''
Step 4: Save the Image and find the previous comic

At this point, th eimage file of the comic is stored in the res variable. You need to write this image data to a file on the hard drive.
You'll need a filename for the local image file to pass to open(). The comic_url will have a value like "http://imgs.xkcd.com/comics/heartbleed_explanation.png" - 
which you might have noticed looks a lot like a file path. And in fact, you can call os.path.basename() with comic_url, and it rill return just the last part of the URL,
"heartbled_explanation.png". You can use this as the filename when saving the iamge to your hard drive. You join this name with the name of your xkcd folder osing os.path.join() so that your program uses backslashes
on Windows and forward slashes on OS X and Linux. Now that you have the filename, you can call open() to open a new file in "wb" or write binary mode.

Remember from earlier in this chapter that to save files you've downloaded using Requests, you need to loop over the return value of the iter_content() method.
The code in the for loop writes out chunks of the image data (at most 100,000 bytes each) to the file and then you close the file. The image is now saved to your hard drive.

Afterward, the selector "a[ref="prev"]" identifies the <a> element with the rel attribute set to prev, and you can use this <a> element href attribute to get the
previous comic's URL, which gets stores in the url variable. Then the while loop begins the entire download process again for this comic.
'''

parser = argparse.ArgumentParser(description="Downloads XKCD comic image and stores them in a folder, starting from newest to oldest.")
# parser.add_argument("echo", help="Echoes whatever you want to say here", type=str)
# parser.add_argument("-L", "limit", action="store_const", help="The number of downloads you want to limit", type=int, default=1)

image_download_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xkcd_comics")

def main():
	url = "http://www.xkcd.com"
	if not os.path.exists(image_download_directory): # store comics in ./xkcd_comics
		os.mkdir(image_download_directory)
	# os.makedirs("xkcd_comics", exist_ok=True) 
	while not url.endswith("#"):
		# TODO: Download the page
		print("Downloading web page: {} ...".format(url))
		res = requests.get(url)
		res.raise_for_status()
		soup = bs(res.text, features="html5lib")

		# TODO: Find the URL of the comic image
		print("Downloading images from {} comic url ...".format(url))
		comic_elements = soup.select("#comic img")
		if comic_elements == []:
			print("Could not find comic image")
		else:
			try:
				comic_url = "http:" + comic_elements[0].get("src")
				print("This is the comic url: {} ...".format(comic_url))
				# TODO: Download the image
				print("Downloading image { ...".format(comic_url))
				res = requests.get(comic_url)
				res.raise_for_status()
			except:
				# skip this comic
				prev_link = soup.select('a[rel="prev"]')[0]
				url = "http://www.xkcd.com" + prev_link.get("href")
				print("This is the link to the previous comic: {}".format(url))
				continue

		# TODO: Save the image to ./xkcd
		read_size = 100000 # 1 Kb
		image_file = open(os.path.join(image_download_directory, os.path.basename(comic_url)), "wb")
		for chunk in res.iter_content(read_size):
			image_file.write(chunk)
		image_file.close()
		# with open(os.path.join("./xkcd_comics", os.path.basename(comic_url)), "wb") as image_file:
		# 	for chunk in res.iter_content(read_size):
		# 		image_file.write(chunk)

		# TODO: Get the prev button's url
		prev_link = soup.select('a[rel="prev"]')[0]
		print("This is the result of the previous link href: {}".format(prev_link.get("href")))
		url = "http://www.xkcd.com" + prev_link.get("href")
		print("This is the new url: {}".format(url))

	print("Done.")

if __name__ == "__main__":
	parser.parse_args()
	main()