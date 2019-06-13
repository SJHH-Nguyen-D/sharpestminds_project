# List comprehension and generator refresher video walkthrough

import random
import numpy as np 

# using a lambad expression for list comprehension
nums = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ]
my_list = [ n**2 for n in nums]
print(my_list)

# using lambda functions with list comprehension
# the map() function runs an iterant from an iterable list through an anonymous lambda function.
# list comprehensions in this case, do away with map functions as they are no longer needed. 
# my_list = map(lambda x: x*x, nums) # this returns a map object

# using conditional statements to create a list comprehension
# Task: n for each n in nums if n is even:
my_list = [ n for n in nums if n%2 == 0 ]
print(my_list)

# using filter and lambda to create a list comprehension
# Instead of using a map and lambda function, we look at the filter function
# the filter function together with the lambda function filters through a list with a specific set of conditions and returns the result
my_list = filter(lambda x: x%2 == 0, nums) # returns an iterable filter object, which is just as readable as a map function
for x in my_list:
	print(x)

letters = 'abcdefghijk'

# create a letter number pair of tuples (number, letter) 
# to create a list from a list comphrension with a nested loop, you just gotta stack for-loops in succession one after another like so:
my_list_of_pairs = [(n, l) for l in letters for n in nums]
print(my_list_of_pairs)
print(len(my_list_of_pairs))

# also a short list example of the above
my_list_of_pairs = [(n, l) for l in 'abcd' for n in range(0,4)]
print(my_list_of_pairs)
print(len(my_list_of_pairs))

# Using dictionary comprehensions
hero_name = ['Batman', 'Iron Man', 'Spider Man'] 
identity = ['Bruce Wayne', 'Tony Stark', 'Peter Parker']

# I want a dict{"name": "identity"} for each name, hero in zip(names, heros)
# first we run the zip() method on the two lists to return a list of paired tuples by each element in each of the lists
my_zip = zip(hero_name, identity) # returns a zip object, in which each element in the object is a tuple of the paired elements of the combined lists
for i in my_zip:
	print(i)

# this is the vanilla way to do it:
my_dict = dict()
for hero, name in zip(hero_name, identity):
	my_dict[name] = hero # one hero name for each identity name
print(my_dict)

# this is how to achieve the same thing but in the form of a dictionary comprehension...much more concise
# this is also a good exampe of how the zip function is made to create a zip object of list pairs to go through
my_dict = {name: hero for name, hero in zip(identity, hero_name)}
print(my_dict)

# Like with the conditional list comprehension, you can also make conditional dictionary comprehensions in the same fashion
# We would like to put the stipulation of creating a dictionary comprehension if the identity name is NOT Peter Parker
my_dict  = {name: hero for name, hero in zip(identity, hero_name) if name != "Peter Parker"}
print(my_dict)

# Using Set Comprehension
# As a refresher, a set is a list of unique values
nums = [1, 2, 3, 4, 4, 4, 4, 4, 3333, 9, 9, 3, 1, 4, 7, 3]
my_set = set(nums) # using the set method on the list to create a set
print(my_set)

# you can create a set comprehension much in the same way you created a dictionary comprehension except you don't need the colon
my_set = {n for n in nums} # as easy as that. Looks like dict comprehension because of the braces, but it is a set object
print(my_set)
print(type(my_set))

# Next we will be looking into generators
# This is the vanilla way to create a generator:
def generator_function(nums):
	for n in nums:
		yield n * n * n # note the keyword yield

nums = [1, 2, 3, 4, 5]
# my_gen = generator_function(nums)
# for i in my_gen:
# 	print(i)

# if you were to create the same thing with a generator comprehension, the syntax would look fairly similar
my_gen = (n*n*n for n in nums)
for i in my_gen:
	print(i)