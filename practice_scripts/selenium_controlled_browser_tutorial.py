'''
The selenium module lets python directly control the browser by programmatically clicking links and filling in login information, alsmost as though
there is a human user interacting with the page. Selenium allows you to interact with web pages in a much more advanced way than Requests and Beautiful Soup.
But because it launches a web browser, it is a bit slower and hard to run in the background, if, say, you just need to download some files from the Web.

Use firefox if you are working with selenium
'''
from selenium import webdriver
browser = webdriver.Firefox()
print(type(browser))

# tar -xzvf file.tar.gz
# export PATH=$PATH:/path/to/directory/of/executable/downloaded/in/previous/step

# # or
# from selenium import webdriver
# driver = webdriver.Firefox(executable_path=r'your\path\geckodriver.exe')
# driver.get('http://inventwithpython.com')