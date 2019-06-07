from bs4 import BeautifulSoup
import urllib3, requests

link = 'https://entomofarms.com/recipes/#our-recipes'

import requests
from bs4 import BeautifulSoup

def get_html(url):
    response = requests.get(url)
    if not response.status_code == 200: 
        print(response.status_code)
    # print(response.text)
    return response.text

html = get_html(link)

def make_soup(url):
  return BeautifulSoup(get_html(url), 'lxml')

soup = make_soup(html)
# print(soup.prettify())

for section in soup.find_all(attrs={"class": "inner-wrap"}):
	print("This is a section: \n{}".format(section.href)

# if __name__ == "__main__":
# 	main()