# 100 Data Science in Python Interview Questions and Answers for 2018



1) How can you build a simple logistic regression model in Python?
```
from scikit learn's linear_models library 
with:

from sklearn.linear_models import LinearRegression
```
2) How can you train and interpret a linear regression model in SciKit learn?
```
You can train one by calling the .fit() method to train the model, the .predict() method to create inferences from the linear model. By calling clf.score(X, y), a coefficient of determination R^2 is returned. The values span -1.0 - 1.0 and indicate fit of the model to the dependent variable. 1.0 represents the best fit, whereas a 0 indicates no fit.
```

3) Name a few libraries in Python used for Data Analysis and Scientific computations.

	NumPy, SciPy, Pandas, SciKit, Matplotlib, Seaborn

4) Which library would you prefer for plotting in Python language: Seaborn or Matplotlib?

	Matplotlib is the python library used for plotting but it needs lot of fine-tuning to ensure that the plots look shiny. Seaborn helps data scientists create statistically and aesthetically appealing meaningful plots. The answer to this question varies based on the requirements for plotting data.

5) What is the main difference between a Pandas series and a single-column DataFrame in Python?
```
a pandas series and a single dataframe column have different methods that are available to them.

Dataframes: 
Two-dimensional size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure

Pandas Series:
Elements or children of the Pandas Dataframe object. A series is the data structure for a single-column of a Dataframe. 
```
6) Write code to sort a DataFrame in Python in descending order.
```
 sorted_dataframe = df.sort_values(['column label'], ascending=[False])
```
7) How can you handle duplicate values in a dataset for a variable in Python?
```
bill_data = pd.read_csv(csvdatafile)
dupes = DataFrame.duplicated()

# look at the rows in the dataframe that are duplicates
print("There are {} duplicates in this dataframe".format(sum(dupes)))

# drop the duplicates
bill_data_uniq=bill_data.drop_duplicates()

# if you are looking for duplicates along the column axis or feature axis, you can do this:
dupe_id=bill_data.feature.duplicated()

# drop the duplicates along a variable/feature
bill_data_cust_uniq=bill_data.drop_duplicates(['cust_id'])
```
8) Which Random Forest parameters can be tuned to enhance the predictive power of the model?
```
we can first we what hyperparameters are available to us from the random forest estimator by doing the following

from sklearn.ensemble import RandomForestRegressor # RandomForestClassifier if you classifying
from pprint import pprint

rfclf = RandomForestRegressor(random_state=42)

print("Paramaters currently in use:\n")
pprint(rfclf.get_params())

* max_depth=The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

* min_samples_split = The minimum number of samples required to split an internal node

* min_samples_leaf = The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression.

* n_estimators = number of trees in the forest

* bootstrap = Whether bootstrap samples are used when building trees. If False, the whole datset is used to build each tree.

* max_features = The number of features to consider when looking for the best split
```
9) Which method in pandas.tools.plotting is used to create scatter plot matrix?
```
pandas.plotting.scatter_matrix
```
10) How can you check if a data set or time series is Random?
```
To check whether a dataset is random or not use the lag plot. If the lag plot for the given dataset does not show any structure then it is random.

this can be done with pandas's innate plotting lag function, which takes as input a series, array or list and plots it against t lags.

from pandas.plotting import lag_plot
from pandas import Series
import matplotlib.pyplot as plt

series = Series.from_csv(datafile, header=0)
lag_plot(series)
plt.show()

BONUS: you can also combine correlations with lag plots to see the strength of association between the current observation and the lagged observation by a plot known as autocorrelation (self-correlation).

Correlations are between -1 and 1. Correlation scores closer to 0 indicate weaker relationships

This can be done in pandas with:

from pandas.plotting import autocorrelation_plot

autocorrelation_plot(series)
plt.show()
```
11) Can we create a DataFrame with multiple data types in Python? If yes, how can you do it?
```
You can specifically cast feature columns using the .astype() method and then specifying the type that you want that column to be casted as.

import panda as pd
import numpy as np

df = pd.read_csv(datafile)
df.numeric_feature.astype(np.float64)

you can do this fo select features.
```
12) Is it possible to plot histogram in Pandas without calling Matplotlib? If yes, then write the code to plot the histogram?
```


```
13) What are the possible ways to load an array from a text data file in Python? How can the efficiency of the code to load data file be improved?

	numpy.loadtxt ()

14) Which is the standard data missing marker used in Pandas?

	NaN

15) Why you should use NumPy arrays instead of nested Python lists?
```

```
16) What is the preferred method to check for an empty array in NumPy?
```

```
17) List down some evaluation metrics for regression problems.
```

```
18) Which Python library would you prefer to use for Data Munging?

	Pandas

19) Write the code to sort an array in NumPy by the nth column?

	Using argsort () function this can be achieved. If there is an array X and you would like to sort the nth column then code for this will be x[x [: n-1].argsort ()]

20) How are NumPy and SciPy related?
```

```
21) Which python library is built on top of matplotlib and Pandas to ease data plotting?

	Seaborn

 22) Which plot will you use to access the uncertainty of a statistic?

	Bootstrap

23) What are some features of Pandas that you like or dislike?

24) Which scientific libraries in SciPy have you worked with in your project?

25) What is pylab?

	A package that combines NumPy, SciPy and Matplotlib into a single namespace.

26) Which python library is used for Machine Learning?

	SciKit-Learn 

27) How can you copy objects in Python?

The functions used to copy objects in Python are-

	Copy.copy () for shallow copy

	Copy.deepcopy () for deep copy

However, it is not possible to copy all objects in Python using these functions.  For instance, dictionaries have a separate copy method whereas sequences in Python have to be copied by ‘Slicing’.

28) What is the difference between tuples and lists in Python?

```
Tuples can be used as keys for dictionaries i.e. they can be hashed. Lists are mutable whereas tuples are immutable - they cannot be changed. Tuples should be used when the order of elements in a sequence matters. For example, set of actions that need to be executed in sequence, geographic locations or list of points on a specific route.

```
29) What is PEP8?
```
PEP8 consists of coding guidelines for Python language so that programmers can write readable code making it easy to use for any other person, later on.
```
30) Is all the memory freed when Python exits?

```
No it is not, because the objects that are referenced from global namespaces of Python modules are not always de-allocated when Python exits.

```
31) What does _init_.py do?

```
_init_.py is an empty py file used for importing a module in a directory. _init_.py provides an easy way to organize the files. If there is a module maindir/subdir/module.py,_init_.py is placed in all the directories so that the module can be imported using the following command-

import  maindir.subdir.module

```
32) What is the different between range () and xrange () functions in Python?

```
range () returns a list whereas xrange () returns an object that acts like an iterator for generating numbers on demand.

```
33) How can you randomize the items of a list in place in Python?

```
Shuffle (lst) can be used for randomizing the items of a list in Python

```
34) What is a pass in Python?

```
Pass in Python signifies a no operation statement indicating that nothing is to be done.

```
35) If you are gives the first and last names of employees, which data type in Python will you use to store them?

```
You can use a list that has first name and last name included in an element or use Dictionary.

```
36) What happens when you execute the statement mango=banana in Python?

```
A name error will occur when this statement is executed in Python.

```
37) Write a sorting algorithm for a numerical dataset in Python. 

```

```
38) Optimize the below python code-

```
word = 'word'

print word.__len__ ()

Answer: print ‘word’._len_ ()

```
39) What is monkey patching in Python?

```
Monkey patching is a technique that helps the programmer to modify or extend other code at runtime. Monkey patching comes handy in testing but it is not a good practice to use it in production environment as debugging the code could become difficult.

```
40) Which tool in Python will you use to find bugs if any?

```
Pylint and Pychecker. Pylint verifies that a module satisfies all the coding standards or not. Pychecker is a static analysis tool that helps find out bugs in the course code.

```
 41) How are arguments passed in Python- by reference or by value?

```
The answer to this question is neither of these because passing semantics in Python are completely different. In all cases, Python passes arguments by value where all values are references to objects.

```
42) You are given a list of N numbers. Create a single list comprehension in Python to create a new list that contains only those values which have even numbers from elements of the list at even indices. For instance if list[4] has an even value the it has be included in the new output list because it has an even index but if list[5] has an even value it should not be included in the list because it is not at an even index.

 [x for x in list [1::2] if x%2 == 0]

The above code will take all the numbers present at even indices and then discard the odd numbers.

43) Explain the usage of decorators.

```
Decorators in Python are used to modify or inject code in functions or classes. Using decorators, you can wrap a class or function method call so that a piece of code can be executed before or after the execution of the original code. Decorators can be used to check for permissions, modify or track the arguments passed to a method, logging the calls to a specific method, etc.

```
44) How can you check whether a pandas data frame is empty or not?

```
The attribute df.empty is used to check whether a data frame is empty or not.

```
45) What will be the output of the below Python code –

def multipliers ():

    return [lambda x: i * x for i in range (4)]

    print [m (2) for m in multipliers ()]

The output for the above code will be [6, 6,6,6]. The reason for this is that because of late binding the value of the variable i is looked up when any of the functions returned by multipliers are called.

46) What do you mean by list comprehension?

The process of creating a list while performing some operation on the data so that it can be accessed using an iterator is referred to as List Comprehension.

Example:

[ord (j) for j in string.ascii_uppercase]

     [65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]

Matser Data Science with Python by working on innovative Data Science Projects in Python

47) What will be the output of the below code

word = ‘aeioubcdfg'

print word [:3] + word [3:]

The output for the above code will be: ‘aeioubcdfg'.

In string slicing when the indices of both the slices collide and a “+” operator is applied on the string it concatenates them.

48)       list= [‘a’,’e’,’i’,’o’,’u’]

print list [8:]

The output for the above code will be an empty list []. Most of the people might confuse the answer with an index error because the code is attempting to access a member in the list whose index exceeds the total number of members in the list. The reason being the code is trying to access the slice of a list at a starting index which is greater than the number of members in the list.

49)       What will be the output of the below code:

def foo (i= []):

    i.append (1)

    return i

>>> foo ()

>>> foo ()

The output for the above code will be-

[1]

[1, 1]

Argument to the function foo is evaluated only once when the function is defined. However, since it is a list, on every all the list is modified by appending a 1 to it.

50) Can the lambda forms in Python contain statements?

```
No, as their syntax is restrcited to single expressions and they are used for creating function objects which are returned at runtime.
```