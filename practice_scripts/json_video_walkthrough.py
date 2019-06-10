import json, webbrowser, os, requests

# Some code to extract the json files data from the web api

# save out the json file

# res = json.load(soource) # if the json file was a straight json
res = json.loads(source) # if the data was read in as a json string

# have a look at the json file
print(json.dumps(res, indent=4))

# looping through the json file to extract key pieces of information and 
# placing it into an empty dictionary


# you can even delete one of the attributes of a dictionary by passing in the del function:
for i in res:
	del i["states"]["abbreviations"]

# see. You will notice that this attribute will be missing in the dump
print(json.dumps(res, indent=4))

usd_rates = {}

for item in res["list"]["resources"]:
	name = item["resources"]["fields"]["name"]
	price = item["resources"]["fields"]["name"]
	usd_rates["name"] = price # sets a price value to each of the names in the dictionary

# see one of the conversion rates from the dictionary
print(usd_rates["USD/EUR"])

# if you want to save the data now, you can use the context manager to do this

res = "Hypothetical dictionary-like json string"
filename = "./data_directory/my_json_file.json"
mode = "w"

with open(filename, mode) as json_file:
	json.dump(res, json_file) # writes it out to a json file
	# json.dumps(res, json_file) # writes it out as a json_string