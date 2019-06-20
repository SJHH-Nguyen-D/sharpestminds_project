# Preparing for a Python Interview: 10 Things you should know with Corey Schafer

Link to the [video](https://www.youtube.com/watch?v=DEwgZNC-KyE&t=5s).

This document will be taking notes from this video of the same topic.

1. Know how to write code on a whiteboard or paper

2. Know basic Python control flow 
By control flow, we mean, how to use for-loops, if-else statements, and things like that. 

3. Be able to discuss how you've used Python
If you haven't been able to use Python for some professional level projects up to this point, and you're stuck on some ideas on applications or projects to create, here are a list of some cool ideas that you can try and implement:
* Webscraping for weather information each day with beautiful soup 
* Or a program that performs system tasks such as cleaning up folders on your computer or moves files around and displays CPU and memory, and stuff along those lines

4. Know how to solve common interview problems
* FizzBuzz
Loop through a range of numbers. If the number is divisible by 3 print Fizz, by 5 print buzz, and divisible by 3 and 5, print fizzbuzz, else return the number itself.

```python3
def fizzbuzzer(reps):
	for i in range(1, reps):
		if i % 3 == 0 and i %5 == 0:
		print("FizzBuzz")
	elif i%3 == 0:
		print("Fizz")
	elif i % 5 == 0:
		print("Buzz")
	else:
		print(i)

fizzbuzzer(21)
```

* Fibonacci Sequences
The idea is that you print out a number where each number is the previous two numbers added together.

```python3
a, b = 0, 1
for i in range(10):
	print(a)
	a, b = b, a + b
```

5. Know basic python data types and when to use them. 