# notes with charlie - june 12, 2019

feature selection and engineering

preparation:
depends  on the type of data:
- dna or genomic data:
	- encode them as edit distance

dfeature engine4ering:
choosing features that you want to use in you model
transforms for your model

feature selection
choose a subset of your features such that most valuable features in terms of what boosts the performance of your model
- check for multilinearity in your data. if you features are highly correlated, sometimes that messes up your model
- somke models, if theyu have high collinearity, that is bad because it double weights those features
- some are more immune to those features. 
- depends on the model
- select features can be done with:
	- decision trees
	- L1 penalty induces sparsity for you data so it acts like a feature selector
	- look at what weights have the heighest values and select those weights wiyth the highest values

- sometimes this is paried with domain expertise
- setup some pipelines for different types of methods and 
- data leakage: introducing bias into the mdoel. its bad practice to introduce your dataset to the test set data.
- whatever preprocessing you do before modelling, 
- so you don't want want to calc std dev and mean to ALL data. just training set. 

if you normalize on your training, you would use the same parameters to transform your test set separately.

- PCA is like data compression. It is more like a data fransformation. it's more to see data visualzation. translate your data into more managemable embedding

- target metrics per dataset
- run some logistic regression on it at first
	- most interpretable

in the case where you are no hitting your metrics,
	- do feature selection so that you can fit a decision tree and then look athte feature importaance and select subsets of your attributes
	- ensembling of models to aggregate and reduce variance.
	- for feature selection, you can literally try each feature. 
	- its good to normalize them so that each feature so that each feature can be weighted equally.

surprising information:
- good to dig into why that is the case. 
- It might be that their domain information actually right. 
- it would be good to find out where the disagreements are as to why there are discrepancies. 
- disagreements with how the data is encoded or stored. 

# Always question the results of your findings and models
# Have buy-in from your direct manager: there is a trade-off between 
# it takes experience of working in an organization: ask your colleagues and coworkers who have more experience what you should do in this situation.
in the real world, it is a lot more nuanced.
- you'd want to empathetic and understanding about the stakeholder and consider their needs and consider that they have their own deadlines

# find a real world messy problem and dataset in order to really get a feel for how you should feature engineer
# you will encounter deeper complexity levels of problemas you become a more senior data scietnist. is becomes more ambiguous what the stakeholders want
# or what they care about. what adds valuet o the organization. what is important. 
# they don't need to know everything about your model. they shouldnt. they just want to relevant parts of your model towards making those decisions. 
# as a junior, you will probably not likely be in the positions of educating seniors. some tinmes a simpler model is better.

if you have the hour for ever single data, you can interpolate between them. 

# parse the actual numbers for the date and pass it to the date time object. if its actually a datetime option, 
# use regex to remove the 00:00:00 portion
# datetime object in pandas NOT linux time. allows you to re-index and interpolate

# learnign to write unit tests:
vsion control git, docker, 
assert statements, except try statements. 

# learning more date structures and algorithms and then doing some problems on leet Code() and hacker rank 
# code reviews
# coding styles, best coding software practices

# Soft-Deliverables:
 - Go on Leet Code and learn about it. for the next time wee meet up for an interview
 - Read the cracking the coding interview
 - elements of programming interview for python
 - Don't spend more than an hour on each problem. don't be too quick to look at the solutions
 - It's like math, you can look at the problem, you can udnerstand the nature of the problem for the next time you encounter it.