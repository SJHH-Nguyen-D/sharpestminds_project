# Answers to common interview questions

############################################ Machine Learning ###########################################################


## explain PCA, LDA, SVD

```

```

## what is feature engineering

```

```


## What is bias in a model?

```
The INABILITY for a machine learning method to capture the RELATIONSHIP BETWEEN THE FEATURES AND THE TARGET is called BIAS.

Straight line:
The straight line has high bias because it has the inability to fit along the relationship curve between the predictor variables and the target variable.

Squiggly Line:
The squiggly line has low bias because it fits really well to the training data points. T
```
## What is variance in a model?

```
The difference in fits BETWEEN DATASETS is called VARIANCE. This can be a comparison between the training dataset and the testing or dev dataset.

Straight line:
The straight line has high bias because it has the inability to fit along the relationship curve between the predictor variables and the target variable. In contrast to the Squiggly Line, it has high bias, since it cannot capture the curve in the relationship between weight and height, however, the straight line has relatively low variance, because the sum of squared residuals (error) are very similar, even for different datasets (comparing the training dataset to the testing dataset). In essence, the straight line might only give good predicts on both datasets, but not great predictions. BUT they will be consistently good predictions. 

Squiggly Line:
The squiggly line has high variance because it results in vastly different sums of squared errors for different datasets(training dataset, testing dataset). That is, it is hard to predict how well the squiggly line will perform with future datasets. It might do well sometime and at other times it, might do terribly. 
```

## Define variance and bias and explain the bias Trade-off

```
Take the example of using weight to determine the height from a dataset of mice.

The first machine learning algorithm we will be looking at is Linear Regression also known as Ordinary Least Squares regression. It has high bias because it cannot possibly completely capture the relationship between two variables, in this example. It cannot bend or conform to a curve, and ultimately remains linear.

The second machine learning algorithm fits a squiggly line to the datapoints. The line is super flexible and hugs the training set or the data points on which it tries to learn the relationship between X and Y on, along the arc of the true relationship. Because the Squiggly line algorithm can handle the arc in the true relationship bewteen the height and weight, it has very little bias. 

We can compare how well the Straight Line and the Squiggly Line fit the training set by calculating the sum of of squares of their errors. The ones that have the least error in their prediction is the most accurate in predicting the target variable, therefore, is more "accurate". In another words, we measure the distance from the FIT LINE to the data, SQUARE THEM (they are squared so that their negative distances do not cancel out the positive distances). We see that the sum of squared residuals between the line of fit for the SQUIGGLY LINE and the data points fits so well that the distances are all 0 from the line, such that the sum of their squared differences are effectively 0 as well. 

Thus, the Squiggly Line wins. 

BUT, so far we have only calculated the sum of squared errors for the TRAINING SET. We also have a TEST SET. Now let's caulcature the sum of squares for the TESTING set. Now it is a contest to see if the Straight Line fits the TESTING SET better than the Squiggly Line. In this case...,

The Straight Line wins.

Even though the squiggly line did a great job at learning the relationship between the features and the target variable, it does a terrible job at this for the testing set. The straight line consistently predicts okay between the training and testing sets, and ultimately shows lower variance between performance between dataset performances, meaning that the squiggly line shows high variance and low bias (remember, bias can also be thought of as the inability for a model to learn the relationship curve between the feature variables and ther target variables.). 

TERMINOLOGY ALERT
Because the Squiggly Line fits the training set really well, but not the testing set, we say that the Squiggly Line is OVERFIT to the training data. In machine learning, the ideal algorithm has low bias, and can accurately model the true relationship AND has low variance or low variability between between predicting on different datasetsk by producing consistent predictions across different datasets. This is done byfinding the sweet spot between a simple model and a model that is too complex. In this case, the complexity of the model can be plotted on the X-axis, and the acccuracy performance of a model can be plotted on the y-axis.

TERMINOLOGY ALERT
Three commonly used methods for finding the sweet spot between a simple and complex models exist:
* cross validation
* Regularization
* Boosting
* Bagging

```

## How do you know you're overfitting

```
Because the Squiggly Line fits the training set really well (HIGH BIAS), but not the testing set (LOW VARIANCE), we say that the Squiggly Line is OVERFIT.
```

## What is cross validation? Why do we do it and how does it help us?

```
Cross validation allows us to compare different machine learning methods and get a sense of how well they work in practice for a particular dataset or problem. 

In machine learning, we:
1. train a model to estimate the parameters of the model that would get us to learn the relationship between the feature variables and the target variable. 
2. evaluate how well the machine learning method works in the testing phase. 

We often use 0.75 of the data to train on and 0.25 to test the data on, however, how do we know these data points that were selected for the evaluation process would be the best ones to train on? What if there were noisy outliers in the initial split and thus, would cause models to underfit due to high variance during the evaluation step. 

This is how cross validation works.
For example, we use the first 0.75 of the data points for training and the 0.25 of the data as the validation set for evaluation. We then record the performance of the model on this split. Then we use the same proportional split of the data, except a different set of data points are selected as the training data points and a different set of dev set data points for evaluation. We then keep track of the performance of the model on that that data train_test_split. In the end, every block of data is used for testing. We can then either average out the results of teh recorded performances or select the best one as the ambassador statistic to compare against other models and how well they perform on this particular dataset/problem.

Also, if you are hyperparamter tuning, you use k-fold cross validation to help you find the best value for this HP.
```


## What is Regularization and how can it help in tuning a model?

```
Regularization is one of three popular methods to reduce overfitting of a model, in which it can help find the sweet spot a simple and a complex model

The process of regularization can also be thought of as desensitization

With regularization, we try to find a new model with new parameters that doesn't overfit to the training data as well.
In other words, we introduce a SMALL AMOUNT OF BIAS into how the NEW LINE is fit to the data. But in return for that small amount of BIAS, we can get a significant DROP IN VARIANCE. Ultimately, this reduces the overfitting on the training set while simultateously providing a better fit on the new testing data. In other words, by starting with a slightly worse fit, (ridge) regression can provide better long term preductions

Regularization helps to do this by introducing a penalty to the model parameters (weights) of the model, and that is multiplied by a lambda coefficient (which determines how severe this penalty is treated), based on some error/cost/loss function.

```

## What is L1-Ridge Regularization/Regression

```
Ridge Regression is also known as L1 regression or L1 regularization. Typically for zero-ing out the learned feature coefficients for a learning model, meaning that by selecting the features that are not zeroed out, and dropping the use of the zeroed out features in the model, we can eliminate some of the complexity of the model to reduce bias in the model so that we can better learn the relationship between the model parameters, the feature variables and the target variable.

When least squares or linear regression determines what values the parameters of the model it should take, it tries to minimize it's loss/cost/error function:

	sum of squared residuals.

In contrast, L1-Ridge regression tries to minimize it's cost function (the curly braces are just there to make the components of the equation more readable). The slope is the same as the weight/parameter coefficient of feature variable: 
	
	{ sum of squared residuals }  + { lambda * slope**2 }

	AKA 

	{ sum of squared residuals }  + { lambda * learned_feature_weight**2 }

	AKA

	{ sum of squared residuals }  + { Ridge Regression Weighted Penalty }

# Lambda
Lambda is a regularization coefficient common to both L1-Ridge Regression and L2-LASSO regression. The {slope*2} component adds a PENALTY to the model parameters in traditional least squares cost function and {lambda} determines how severe the penalty is. Lambda can take on any value from 0 to positive infinity. If lambda is 0, then the ridge regression penalty is also 0. As we increase the the regularization parameter, we increase the effect of the penalty parameter, and thus, the line because less and less sensitive to the effects of the feature inputs. If lambda is infinite, then we are left with a horizontal line, not fitting to any of the data. 

How do we decide what value to give lambda? We just try a bunch of values for lambdaand use cross-validation, typically 10 fold cross validation, to determine which ones result in the LOWEST VARIANCE between training and dev datasets.


# Ridge Regression Penalty
In this example, the {sum of squared residuals} for the *OLS regression line* is 1.69, and for the *L1-Ridge Regression line*, the {sum of squared residuals + L1-Ridge Regression weighted Penalty}  is 0.74. Thus if we wanted to minimize the sum of the squared residuals + the Ridge Regression weighted penalty, we would choose the L1-Ridge Regression line over the OLS regression line. 

WITHOUT the SMALL AMOUNT OF BIAS that the penalty creates, the OLS regression line has a LARGE AMOUNT OF VARIANCE. In contrast, the L1-Ridge Regression Line, which has a SMALL AMOUNT OF BIAS due to the RIDGE REGRESSION WEIGHTED PENALTY, has LESS VARIANCE.


Before we talk about lambda, we have to talk about the effect of the l1-ridge regression penalty has on how the line is fit to the data. When the SLOPE OF THE LINE(learned weight/coefficient of feature variable) IS STEEP, then the PREDICTION FOR THE TARGET VARIABLE IS VERY SENSITIVE to SMALL CHANGES IN THE FEATURE VALUE. By the same measure, when SLOPE IS SMALL, then for every one unit increase in the input feature value, then the prediction for the target variable barely increases.

In the previous example, the ridge regression line resulted in a line with a smaller slope, which means that predictions made by the ridge regression line, are less sensitive to the feature values for the 'weight' attribute, than the OLS regression model. 


Ridge Regularization also works when we deal with discrete categorical variables. The slope or feature weight in this case would be equivalent to the difference between the average target attribute value for feature1-group1 vs the average target attribute value for feature1-group2. They can be thought of a binary values. Groups that are assigned the 0 value rely on other terms such as the y-intercept to predict their target value alone. For the Groups that are assigned to the 1 value, they rely on both the y-intercept and the slope/feature weight to determine their target attribute value. 

The whole point of doing ridge regression is because small sample sizes like these can lead to poor least squares estimates that result in terrible machine learning predictions. 

You can even use ridge regression for logistic regression problems and the ridge regression function would look like this instead:

	{sum of squared likelihoods} + {lambda * slope**2}

It helps predict whether or not a particular class label is sensitive to the feature variables. When applied to logistic regression, ridge regression optimizes the sum of the likelihoods as opposed to squared residuals in OLS regression because logistic regression is solbed using Maximum Likelihood.

##### Ridge Regression for Regularization in Complicated models #####
Now we've seen how using ridge regression has been used to reduce variance by shrinking parametersand making our predictions less sensitive to them. We can apply ridge regression to complicated models as well such as multidimensional or multivariate datasets. In general the ridge regression penalty contains all the parameters/feature weights except for the y-intercept. 

An equation for a ridge regression could look like this for a complicated model:

	lambda * {slope**2 + diet_difference**2 + astrological_effect**2 + airspeed_scalar**2}

Every parameter except for the y-intercept is scaled by the measurements. 

That said, we need at least two data points for a 1-feature-1-target dataset to even have estimate the parameters of the data. We need at least three data points if we had a 2-feature-1-target dataset.

The rule is then that you require (F+1) data points, where F is the number of variates/features in your dataset.

Having enough datapoints-to-features in a dataset for a large parameter dataset might be bonkers and ridiculous to achieve. But we can solve for this using RIDGE REGRESSION. 

It turns out, we can solve for a large feature dataset with less than the required number of samples, using RIDGE REGRESSION. Ridge regression can find the solution with cross validation and ridge regression penalty that favors smaller parameter values.

Summary:
1. when teh samples sizes are realtively small , the ridge regression can improve predictions made from new data(ie. reduce variance) by making the predictions less sensitive to the training data. this is done by adding a ridge regression penalty to the thing tha tmust be minimized. 

2. lambda is determined using cross validation. 

3. even when there isn't enough data to find the OLS parameterestimates, ridge regression can still find a solution with cross validation and ridge regression penalty.

Another note, after having read through L2-Lasso regression, is that Ridge regression tends to do a little better than L2-LASSO regression when most of the variables are useful. When there are lot of useless variables in the model, you get a lot of variance as a result.
```

## What is L2-LASSO Regression/Regularization
``````
Read Ridge Regression.

L2-LASSO regression is SIMILAR to ridge regression in these manners:

1. Adds a penalty to the parameters/weights/coefficients of the model with the goal of making the model less sensitive to the values training data. This is particularly useful when you have a small amount of training data to work with. Larger coefficients are associated with larger penalties. In this regard, you introduce a small amount of bias into the your data with the goal of reducing the amount of variance that your mdoel has in terms of its fit from the training data vs. the dev/testing data.
2. LASSO also has a lambda parameter which is the regularization parameter. It can be 0 to positive infinity and it determines the strength of effect of the L2-LASSO regularization penalty. Just like L1-Ridge Regression, the lambda for L2-regression can be discovered using cross-validation
3. Just like L1-ridge regression, they can be applied to complicated models that combine discrete and numeric/continuous data types. 
4. just lihe L1-ridge regression, L2-LASSO regression contains all of the estimated parameter/learned_weight_coefficients in the calculation of the penalty, except for the y-intercept.
5. When lambda is 0, there is no regularization and no penalization of the model coefficients, as the penalty has been zero'd out by the lamda, thus the model is essentially just minimizing the sum of squared residuals. 
6. The penalty does not affect all the parameters of the model all the same.


L2-LASSO regression is similar to ridge regression except for a few important DIFFERENTiations.

1. it is not differentiable. It is the absolute shrinkage selection operator after all
2. you take the absolute value of the slope instead of the squared value of the slope/parameter/learned_feature_weights
3. The big difference between L1-ridge regression (which uses the coefficients**2) and L2-LASSO regression (which uses the absolute value of the coefficients), is that L1-Ridge regression can only shrink the slop asymptotically close to 0, whereas L2-LASSO regression can shrink the slope ALL THE WAY to 0. 
4. For ridge, no matter how small you make lambda, you can't shrink the coefficients of the features down to zero. However, for LASSO regression, you can shrink the learned parameters down to 0. This can be used as a method to do feature selection/ and reduce overfitting at the same time. It is a little bit better than ridge regression in this regard at reducing the variance in models that contain a lot of useless variables. This makes the final equation easier to interpret. 
``````

## What is Boosting and how can it help in tuning a model?

```
The StatQuest video on XGBoost, Gradient Boosting Classifiers, AdaBoost show an example of boosting in action
```
## What is Bagging and how can it help in tuning a model?

```
The StatQuest video on Random Forest show bagging in action
```

## what is feature selection. what are some methods for doing this.

```

```

## what are eigen values and eigen vectors

```

```

## what is covariance? How is it related to correlation

```

```

## What is the difference between a classification problem and a regression problem?

```

```

## Pick a machine learning algorithm and explain how it works.

```

```

## what is entropy

```

```

## what is data augmentation? 

```

```

## what is expectation maximization

```

```

## what is convex hull optimization

```

```

## what is a manifold?

```

```

## what is the loss function for a regression problem. what is are regression problems trying to optimize or reduce for? How is it calculated?

```

```



############################################### Statistics #################################################################

## Explain recall and precision

```

```

## what is a f1 score

```

```

## what is support

```

```

## Statistics: what is type 1 and type 2 error

```

```

## What is a random variable and what is an example
```
Random variables are ways to map random processes to numbers. So if you have a random process such as flipping a coin or rolling dice or measuring the rain that might fall tomorrow, you are really just mapping outcomes of that to numbers. you are quantifying the outcome.

What is a random variable X: 1 = heads, 0 = tails. 

what is a random variable Y: sum of the upward face after rolling 7 dice. 

As soon as you quantify outcomes, you can start to do a little bit more math on these outcomes, and you can start to use a little bit more mathematical notation on the outcome. If you cared about the probability that the sum of the upward facing rolled dice is =< 30, the old way of denoting this is

P(sum of upward facing dice after rolling 7 dice <=30)

But now we can write:

P(Y <=30)

now if someone wanted to express whether the sum of the upward facing 7 dice was even, they could now write it as such.
P(Y%2==0)

Note that these random variables differ from traditional variable that you would see in your algebra class. With traditional variables, you can solve them for them or assign them. That is not the case for a random variable. A random variable can take on many many different values with different probabilities and so it makes a lot more sense to talk about the probability variable equalling a probability.

A random variable can take on discrete or continuous values such that they are denoted as discrete random variables and continuous random variables. Continuous random variables can take on any value within a range or interval and that range can even be infinite. Discrete random variables take on district/separate values such as heads or tails or 0 or 1. Thing of Discrete random variables as class labels in a classification problem in a machine learning task. A good indication that you are dealing with DRVs is if you can count the values that it can take on. The exact mass of a random animal selected at the New Orleans Zoo. DRV's can also be years. This is so because they are a (most likely finite, but it does not have to be a finite number of values) number of COUNTABLE values. These are typically states that can have frequency counts. 


```

## what is a probability distribution and what is an example

```
Let's say we make X as the number of heads we get after 3 flips of a fair coin as:

P(X)

A probability distribution is the probability of the different possible outcomes or the difrerent possible values for this random variable. we will plot them to see how this distribution is spread out among those possible outcomes. let's think of 

```

## what is a probability density function

```
# Continuous Random Variable (CRV)
Y = exact amount of rain tomorrow

P(Y) is our probability function.

We can ask ourself, what is the probability that we get exactly 2 inches of rain tomorrow, which can be phrased as such:

P(Y=2)

when we look at graph, we might be tempted to think that the probability that the rain is exactly 2 inches tomorrow is 0.5, but that's the wrong way to think about CRVs. There's atoms involved. Theres an infinity of values between 0.01 and 0.02, so this is the wrong way to think about it.

We can instead then say, what is the probability that Y is almost 2. That is the absolute value of Y-2:

P(|Y-2|<0.1) where 0.1 is a threshold or tolerance value to be considered in included in this probability evaluation. 

The above is equavalent also to this expression: 

P(1.9 <= Y <= 2.1)

So now, we are interested in all values of Y under the area of the probability distribution curve within the range of P(|Y-2|<0.1). The area under this cureve for this range of space would essentially be the definite integral of the probabilty density function/distribution curve, within the specified range. The area under this graph would be the integral from 1.9 to 2.1 of f(x)dx where f(x) is the function for this PDF. The probabiltiy of all the events combined cannot exceed 1.0 or 100%. Thefore, the entire area under the curve = 1.0. 

# Discrete Random Variable (DRV)
By the same measure, this principle that the area under the curve, AKA the sum of all the probabilities of events cannot exceed 1.0 holds true. 

PDFs are how we can think of continuous random variables.


```
## what is the difference between a  probability distribution function and a probability density function?

```
* Probability distribution function and probability density function are functions defined over the sample space, to assign the relevant probability value to each element.

* Probability distribution functions are defined for the discrete random variables while probability density functions are defined for the continuous random variables.

* Distribution of probability values (i.e. probability distributions) are best portrayed by the probability density function and the probability distribution function

* The probability distribution function can be represented as values in a table, but that is not possible for the probability density function because the variable is continuous

* When plotted, the probability distribution function gives a bar plot while the probability density function gives a curve

* The height/length of the bars of the probability distribution function must add to 1 while the area under the curve of the probability density function must add to 1

* In both cases, all the values of the function must be non-negative
```

## what is a cumulative probability distribution function?

```

```

## what is kernel density estimation function

```

```

## what are the Kolmogorov Axioms

```

```

## what is multicollinearity. Why is it bad? How do you deal with it? How do you test for it?

```

```

## what is homoscedasticity. Why is it bad? How do you deal with it? How do you test for it?

```

```

## What is _ 