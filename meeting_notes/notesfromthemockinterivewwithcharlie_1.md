# Notes from the mock interview

define these
random variable
probability distribution

l1 and l2 regularization

bias and variance explanation

how to address multicollinearity, homoscedascity

Review:

says well:

## Behaviour questions:
A little bit on the long side. For the first HR recruiter on screen. Don't want to spend a lot of time on a question. Err on the side of too short as oppose to too long. A good technique is opt-in points.

What is your background in ML:
	* have some internships, projects, courses, hackathons
	* have points where you pause and wait to see if they ask further questions.
	* hit the strongest parts of your background for the story that you want to tell
	* keep it under 30 seconds probably. one or two sentences woudl be ideal for each question

* sound more confident. Take more time to think about what kind of things that question is trying to sus out.
If oyu want sometime to think about thwe question before answer. Say something like: "That's a really interesting question. Give me a second to think about it."

"Have you ever worked in a group setting" what really asking for "Can you work well with others? are you receptive to feedback? Can you deal with conflict. THink of a story to demonstrate you have good qualities with working with others."

Try to have some specific stories or examples ready. Have an elevator pitch at the ready where you can just bust out. What matters a lot is confidence and how you deliver it. It leaks into the emotional state of the interviewer. 

Have your behavioural questions pretty much scripted with all your stories. Have the main points you want to use for each question. Use the technqiue that you can template out for each question.

## Technical Questions:

fundamental concepts of machine learning and statistics. 
* random variables -> stochastic mapping from a probability space to an event space. Coin toss or dice roll. it is not deterministic. It follows some probabilty distribution to some out come. A continuous random variable could be height or weight. Look up on this online. Tie these concepts to other related concepts. A random variable is tie to some probabilty distribution. Each toss of a coin is an event and this random variable characterizes what this event would be be given the rpobability distribution.

bayes theerom conditional probabilities, correlation, independence, two variables being correlated does not imply that they are indepdendendt, but two independent implies taht they are uncorrelated. 

conditional independence. Conditional probability. Frequentist statistics. 

Probability distribution. Integrates to 1. All probabilty distributions are greater than 0. The sum of all probability distributions add up to one. Cohle Magoro Axioms. They are all mutuallty exclusive so you can union them and sum them.

Choosing models part is fine. Becareful about saying independent variables. use something like predictor variables.

The difference between regression and clasification is continuous vs discrete output.

Choosing features is okay. Feature engineering. Imputing mmissing values. Scaling and normalizing if the featueres have similar scale that might matter for similar scale. PCA transformations.. you can also use this to deal with multicollinearity. PCA is also good for high level feature visualization. 

Evaluating a model. Accuracy vs error. accuracy you can think of as how many you get right, vs how many you get wrong. you can look at F1 score, sensitivity (TP over all predicted positives) vs specificity (TN over all predicted negatives). type 1 error vs type 2 error. AUC.

Bootstrap aggregation. Jackknife sampling techniques. Taking subsampling of the data to augment the data. Augmenting data or bootstrap. 

Overfitting. Regularization. Feature selection, reducing the number of features (it can reducd overfitting in a sense), reducing the complexity or capacity of a model. If NN reduce the number of hidden units, layers, runs. If tree use early stopping or pruning. Try a simpler model. If that is a decrease in accuracy, you want to use weight penalties (penalizes the magnitude . large coefficients indicate overfitting. l1 vs l2. L1 is an absolute function(not differentiable. its linear. feature selection. penalizes everything the same). L2 is a squaring function (penalizes quadratically, for very high weights get penalized a lot. penalizes outliers very hard). L1 is good because it induces sparcity. L2 is differential. Google L1 vs L2 ball.) Use dropout or batch norm.

Bias and Variance Tradeoff. Review this!
Deals with complexity of model and the no free lunch theorem. Very important concept.

Choose a model that you really know in and out. Pick the core ML algorithms (logistic regression, decision trees, linear regression, MLP, adaboost, SVM, XGBoost). Naive Bayes, K-Means. Try to code one of these thigns from scratch to show how you know how one of these things works. It's just good practive to learn this coding wise. Read some paper and implement the core algorithm and pass all the tests. just using numpy

Know the assumptions of linear regression. There is a linear relationship between the variables, the number of variables is linear in terms of the parameters. Equal variance along the whole range. The is drawn form a normal distribution and you check that by looking at the residuals and seeing if they are constant. Minimizes the mean squared error or the root mean squared error. The analytical way gets an exact solution(OLS. you have to take the inverse of a big matrix. for somethings you don't have an inverse). Gradient based way is the more preferred method. it is a convex function so you will get to some global optimum. 

Deep learning. works based on back propagation using the chain rule and optimize it using some gradient based methods and doesnt always result in a global optimization. Works with high dimensional data. Works with a lot of compute power. Learn up this 


Deliverables:

read up the resources. that he gave you:
https://www.deeplearingbook.org/contexts/ml.html
https://www.deeplearingbook.org/contexts/prob.html
https://www.deeplearingbook.org/contexts/mlp.html

Come up with the script for behavioural questions

Deploy model for simple rest API

Behavioural interview...think about scheduling one.
