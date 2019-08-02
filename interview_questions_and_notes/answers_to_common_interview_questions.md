# Answers to common interview questions

############################################ Machine Learning ###########################################################

## batch norm

## dropout

## RNNS vs CNN

## how to deal with missing data.

## when to use Linear Regression vs Nueral Network


## K-Means vs KNN


```
K-Means is unsupervised and KNN is supervised.

# K-Means 
* does clustering which can be done on a:
	* line
	* scatter plot
	* heat map

* will also pick the best value for the number K (spoiler, using the elbow method that displays the most variation). 
```

## How does SVM work

```

```


## Define AUC ROC, sensitivity, specificity, recall, precision, F1 score, F2 score, accuracy and when you would want to use each?

```
# ROC and AUC

Remember: a logistic regression curve is a sigmoidal curve (0, 1).

When doing logistic regression for a binary class label, the y-axis is converted to the probability that a particular observation with value along the x-axis being in a particular class. There are only two values notched on the ticks along the y-axis and they are 0 and 1 to indicate the two different class labels. The logistic regression tells us the PROBABILITY that an observation is of a particular class, based on it's x-axis value. However, if we want to CLASSIFY, with DISCRETE CATEGORICAL PREDICTIONS, whether an observation is of a particular class, we need a way to map probabilities to those discrete class labels. 

One way to classify is to set a threshold of probabilities. 

Using a confusion matrix of the classification (the top is actual, the horizontal is predicted). Calculate sensitivity and specificity for each. 

We can repeat this process for each threshold for building the confusion matrix again. This would result in different TP, TN, FP, FN, sensitivity, and specificity. 

If the idea of classifying with a different threshold than 0.5, throws you off, then consider the nature of the classification task. What if it is absolutely imperative that you correctly classify every sample as one particular class label, due to the real world implications that it was incorrectly classified ? (e.g., correctly classifying an individual as having ebola or not). This would mean lowering the threshold, even if it means the result would mean more false positives. 

Thus, the threshold could be set to anything, between 0 and 1. how do we determine which threshold is the best threshold for our use case? We don't actually have to test every single option for thresholds, as there will be some overlaps in creating the same confusion matrix over and over again for some configuration of thresholds. Even if we made a confusion matrix for each one that mattered, it would result in a large numbe rof confusion matrices. 

Instead of being overwhelmed with the sheer number of created confusion matricecs, ROC graphs or plots provide a simple way to summarize all of the information at each threshold. 

ROC graphs are always depicted as such: y-axis is TPR (true positive rate aka sensitivity) and the X-axis is (1-specificity). 1-Specificity is also known as the false positive rate. 

	sensitivity = TP/all real positives

	sensitivity = TP/TP+FN

	specificity = TN/all real negatives

	specificity = TN/TN+FP and therefore,

	FP ratio/rate = 1-specificity


We can look at the ROC curve as a trade off between making a false positive guess against being able to correctly classify true real positives, at each rate of each (0-100%), at each threshold of tradeoff. 

If AUC the ROC curve is 1, then we have a model that can correctly classify all samples at all thresholds(?). The way I think the AUC metric works is that lxw of a square = area. if you are were to form a perfect box of TPR=1 and FPR = 1; FPR*TPR=1.0 or 100%, then we have a perfect correlation between the two. I think, then that the AUC is a generalization or somewhat 'average' of TPR vs FPR, and that the greater the 'lift' in the TPR, the better the model is and the greater the AUC is. 

If we set the threshold to 1.0, we can correctly predict, with 100% correctness all of the TPs but have 100% FPs. That means that we failed to correctly classify the other label. We draw a line from the origin of the graph diagonally to divide the graph. Any point on this line means that th eproportion of correctly classified points are the same as the proportion of incorrectly classified samples. This means that along this line, this is the worst that the model can do to classify because at that point, it is just a random guess as to what sample is classified as what. This diagonal could be thought of as the null model or the worst model. 

From what we can gleen from looking at the ROC curve, the higher we can pull the hump of the ROC curve towards to the top left corner of the graph (FPR, TPR) to (0, 1), then the 'better' the model is at classifying things as True positives WITHOUT any false positives. 

Ultimately, the ROC curves make it easy to identify the best threshold for making a decision.
Thusly, the ROC graph is a higher level visual depiction of what happens to the classification of a binary classification problem, when you adjust the thresholds (with implications on emphasizing the importance of predicting things as one class than the other) for classifying a negative case vs a positive case. We connect the lines to see the ROC curve for each TPR and FPR coordinate pair at each threshold. 

# AUC of ROC
The AUC makes it easier to compare one ROC curve to another. The greater the number, the better the classifier. 

# Precision
The 'P' in precision helps you remember that precision deals with the positive case predictions in a classification problem. Although ROC graphs are often drawn with TPR and FPR to summarize the confusion matrix, there are other metrics that attempt to do the same thing and precision is one of them. Thusly, on the ROC curve, we can replace the FPR along the x-axis with precision. Precision is thus, the proportion of positive results that were correctly classified among all negative case predictions.

	Precision = TP/(all guessed positives)

	Precision = TP/TP+FP

# When is it useful to use Precision over FPR?

We would use precision over FPR when the proportion of positive cases relative to the number of negative cases were much greater, as most of the guesses would be positive anyways. This is because the precision calculation does NOT include the number of TNs in its calculation, and there for is not effected by the imbalance of less negatives than positives. In practice, this sort of imbalance occurs when studying an uncommon phenomenon (such as disease, where there are often more healthy cases than not).




# Recall
Recall starts with 'R' and not 'P' so therefore it cannot be remembered as precision, and thus it is the other metric knonw as recall. The opposite of precision is recall. Recall is the proportion of negative results correctly classified among the negative case predictions.

	Recall = TN/(all guessed negatives)

	Recall = TN/(TN+FN)

# When is it useful to use Recall over FPR? 

We would use recall over FPR when the proportion of negative cases relative to the number of positive cases were much greater, as most of the guesses would be negative anyways. This is because the recall calculation does NOT include the number of TP in its calculation, and there for is not effected by the imbalance of less positives than negatives.


# F1-Score
F1-score is the weighted average of precision and recall. Therefore, this score takes false predictions into account (FN and FPs). Intuitively, it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if FPs and FNs have similar cost. If the cost of FPs and FNs were different, it's better to look at both Precision(deals with FPs) and Recall (deals with FNs).

	F1-Score = 2*Recall*Precision/(Recall + Precision)

```

## What is T-SNE? Explain it.

https://www.youtube.com/watch?v=NEaUSP4YerM

```
student T-Stochastic Neighbors Embedding. TSNE is used for dimensionality reduction as well as visualization technique. It takes a high dimensional dataset and visualizes on a lower dimensional graph, while retaining a lot of the original information such as the similarity between data points. We can see clustering of the data points the more similar they are or how dissimilar they are to different clusters.

Let's take an example of taking a two dimensional scatter plot of points, and then transforming the data and mapping them onto a lower dimensional, flat 1-D plot on a number line. This is a super simple example to explain the concepts behind TSNE so that when we see it applied to a larger dataset. Note that if just projected the points onto the line, we get a big mess, as this doesn't preserve the original clustering from the higher dimensional space. 

At each step of the mapping of a from the 2d plot onto a 1d line, a point is attracted to points it was near to in the higher dimensional scatter plot, and repelled by points it is far from... 

This is how it does what it does.

# Determining the unscaled similarity between one particular point, wrt to the rest of the points
1. determine the similarity between all the points in the scatter plot. For this demonstration, let's just focus on determining the similarity between one particular point and the it's similarity between it and the rest of the points.
	* measure the distance between two points.
2. Plot this distance along a normal or gaussian curve that is centred upon the point of interest
3. Draw a line from this point along the axis to the curve. This line is the unscaled similarity. 
4. We plot the distance from this one point to all the other points, and measure the unscaled similarity (distance from the point on the axis to the normal curve itself) Using a normal distribution means that very distance points have very low unscaled similarity values. By the same measure, more similar points have a smaller distance that are measured from the 2D plot, and thus have greater unscaled similarity values. 
5. Now, we scale the unscale similarity scores so that they add up to 1 (normalize). Why do they have to add up to 1? The width of the normal curve depends on the density of data near the point of interest in the center. The denser the data are around the centre, the higher the peak of the normal is (imagine packing a sand hill and squishing the sand into the centre), and the narrower the base of the peak is. The wider the peak, the greater the standard deviation is. Scaling the unscaled similarity scores will make them the same for both clusters. 
	
	Scaling the similarity values so that they sum to 1:

	(unscaled similarity score score for one point)/(sum of all unscaled similarity scores) = scaled score for one point

T-SNE has a perplexity parameter equal to the expected density. This comes into play later on. 

6. We repeat this process for each of the points on the scatter point wrt to the rest of the other points.

Note: since the width of the distribution depends on the density of distance values surrounding the data points, the similarity score for a particular node, the similarity score for point A to point B might not be the same as the similarity score from point B to point A. So, what is happening here is that t-SNE averages the two similarity scores from the two directions. 

Ultimately, you end up with a matrix of similarity scores. TSNE knows that a cluster cannot be similar to itself so it defines a similarity of a point to itself as 0. 

After we have done all that, NOW, we randomly project the points from the scatter plot onto the number line and calculate scaled similarity scores for the points on the number line. Just like before, that means picking a point, measuring a distance, and lastly, drawing a line from the point to a curve. However, this time, we're using a t-distribution. A t-distribution is a lot like a normal distribution, except the t-distribution isn't as tall in the centre as the normal distribution, however the tail in the t-distribution are taller at the ends. The t-distribution is the t- in t-SNE. 

Like before, we produce a matrix of similarity scores by calculating the 1st) the unscaled similarity scores and then 2nd) scale them. However, unlike the original matrix, this matrix is a mess compared to the one that used the normal distribution. T-SNE works one step at a time, moving a point along the number line so that the t-distribution of similarity scores matrix looks like the normal distribution similarity score matrix. 

# This is the rationale as to why the t-distribution is used:
Without the t-distribution, the clusters would all clump up in the middle and be harder to visualize.

```

## What is LDA and how can I use it?

```
Linear Discriminant Analysis is like PCA (in that it reduces dimensions to a lower dimension) but it focuses on maximizing the separability among KNOWN categories.

What is the best way to reduce the dimensions? A bad way is to just ignore a feature and just map the points onto the one feature and onto the x-axis number line. this is bad because you lose the information of spacial clustering. LDA provides a better way.

For the example of reducing a 2 dimensional scatter plot down to a 1-D line, LDA uses information from both features to create a new axis in a way to maximize the separation between two categories. LDA projects the data onto this new axis in a way to maximize the separation between the two categories/class labels/known clusters. 

How does it do this?
1) Maximize the distance between mean measurements of each category/group
2) Minimize the variation (which LDA calls "scatter" and is represented by s**2) WITHIN each category.
3) we consider both these criteria simultaneously in this expression:

	(u1-u2)**2/ (s1**2 + s2**2)

	or 

	distance**2/(s1**2 + s2**2)

	Ideally Large/ Ideally Small

The reason we want this is because we want the big numerator (large difference of means) to convey a larger distance between categories (smaller scatter within categories means a tighter cluster and tighter similarities between data points), which indicates distinct maximimization of distance between the two categorical groups.

We show an example of why scatter and distance are important for visually understanding the separation of our categories and reducing our dimensions while retaining the spatial relation information between clusters. 

If we only separate the distance between means, we may get clumping in the middle between the two separate categories....HOWEVER, if we optimize the distance between means AND the scatter/variance within categories....then we get nice separation. 

## What if we have 3 features?
The process is the same. We create a new axis that maximizes the distance between the means for the two categories while minimizing the scatter. Then we project the data onto the new axis. This new axis was chosen to maximize the distance between the means of those 3 features (probably scaled) while minimizing the scatter. 

## what if we had 3 categories?

If we had three categories, some small things change. We change the way we calculate the distance among the means. 
1) we find a point that is central to all the data
2) we measure the distances between that main central point (centroid) and a central point to each category (those are considered the means) 
3) now we want to maximize the distance between each category and the central point, while minimizing the scatter within each respective category. 

	(d1**2 + d2**2 + d3**2)/(s1**2+s2**2+s3**2)

The second difference with 3 categories is that there are two axes that are created to separate the data as oppose to the previous 1 axis. That is because the three central points to the category define a plane. This plane is optimize the separation between clusters of categories. 

What if we had MORE features? Suddenly, being able to create 2 axes that maximize the separation of three categories is super cool! It is better than drawing a 10,000 dimensions...

##### LDA compared with PCA ###

# Difference
PCA doesn't separate the categories nearly as well as LDA. HOWEVER, PCA wasn't even trying to separate these categories, It was just trying to look for the genes with the most variation. 

# Similarities
* both rank the new axes in order of importance
* PC1 accounts for the most variation in the data and PC2 does the second best job
* LD1 (the first new axis that LDA creates) accounts for the most variation between categories. 
* Both methods allow you to dig in or drill down to see which features are driving the creation of the new axes (loading scores from PCA and in LDA we can see which features correlate with the new axes)

Conclusion:
* LDA is like PCA - both try to reduce dimensions
* PCA looks at the features with the most variation 
* LDA tries to maximize the separation of known categories. 
```


## What is KL divergence and what does it do?

```

```

## What is dimensionality reduction? How can you go about using dimensionality reduction? 

https://www.youtube.com/watch?v=YaKMeAlHgqQ
https://www.youtube.com/watch?v=ioXKxulmwVQ

```
It is reducing the number of features to reduce the variance in the model, as well as the signal to noise ratio of modelling your features to your target.

There are several approaches that can be done through this.

# PCA

PCA: for visualization and DimRed; you can use the selected principal components to model a classification or regression problem from, and it is an exploratory type of modeling that can be done prior to see the PCs of maximum variance and explainability in your data.

# Percent Missing Values
	* % missing values
	* % records with missing values/ % of total recordfs
	* create binary indicators to denote missing values (or non missing values)
		* This might be a good feature that is engineered
		* you can also use a classifier (simple tree) to figure out how and what values could be classified as missing data or use a nearest neighbors alogorithm to do it
	* review or visualize variables with high % of missing values

	```python3
	

	```

# Amount of variation
	* Drop features or review variables that have a very low 	variation
	* variance(x) = sigma **2  = std dev **2
	* either standardize all variables, or use standard deviation sigma to account for variables with different scales
	* drop variables with zero variance (unary)
	
	```python3


	```


# Pairwise Correlation
	* many variables are often correlateed with one another, and hence redundant
	* if two variables are highly correlated, keeping only one will help reduce dimensionality reduction without too much loss of information
		* which variable do you keep? The one that has a higher correlation coefficient with the target.

	```python3
	

	```

# Mutlicollinearity
	* when two or more variables are highly correlated with one another
	* dropping one or more variables should help reduce dimensionality wihtout a substantial loss of information 
	* which variables to drop? Use Condition Index

	```python3
	

	```

# Cluster Analysis
	* dimensionality reduction techniques emphasize similarity and correlation
		* identify groups of variables that are as correlated as possible among themselvesand as uncorrelated as possible with variables in other clusters.
	* reduces multicollinearity - explicability is not always compromised
	* when to use?
		* excessive multicollinearity
		* explanation of the predictors is important

	```python3
	from sklearn.cluster import FeatureAgglomeration
	varclus = FeatureAgglomeration(n_clusters=10)
	varclus.fit(train_scaled)
	train_varclus = varclus.transform(train_scaled)

	```
	This retuns a polled value for each cluster (i.e., the centroids) which can be used to build a supervised learning model


# Correlation with the Target
	* drop variables taht ahve a very low correlation with the target
	* if a variable has a very low correlation with the target it's not going to be useful for the model (prediction).

	```python3
	
	absolutecorrelationwithtarget = list()

	for var in df.columns:
	absolutecorrelationwithtarget.append(abs(y.corr(X[var])))

	# where df.columns are all the variables
	```

# Forward Selection
	1. Identify the best variable (e.g., based on model accuracy)
	2. Add the next best variable into the model
	3. And so on until some predefined criteria is satisfied

	```python3
	loop through every feature of the model and use it as a the predictor variable and keep only the features that result in the greatest sequential increase in model performance and as little information loss as possible
	```

# Backward Selection
	1. start with all variables included in the model
	2. drop the least useful variables (e.g., based on the smallest drop in model accuracy or greatest rise in model accuracy)
	3. And so on until some predefined criteria is satisfied

	```python3
	loop through all features of the model and iteratively remove features that result in a better result than before

	```

# Stepwise Selection
	* similar to forward selection process but a variable can also be dropped if it's deemed not as useful anymore after a certain number of steps

	```python3
	loop through all features of your model and iteratively add or remove the feature to the model

	```
# LASSO
	* Has been used for linear models in regularization, however, there is regularized logisticregression, which works the same way
	* feature selection and regularization in one go

	```python3

	while jLASSO <= maxpreds:
		thislassofit = LogisticRegression(
			C=ThisC,
			penalty='l1',
			random_state=123
			).fit(train1_scaled, y_train)

		modelcoeff = np.transpose(thislassofit.coef_)[np.where(thislassofit.coef_ <> 0)[1]]

		thislassopreds = train.keys()[np.where(thislassofit.coef_ <> 0)[1]]

		thisparamslist = pd.DataFrame(zip(thislassopreds, modelcoef))

	```
# Tree-Based Models
	* forest of trees to evaluate the importance of features
	* fit a number of randomized decision trees on various SUB-SAMPLES of the dataset and use averaging to rank order features by their importance contribution. You can use this with XGBoost or Random Forest

	```python3
	ordered_params["DTree"] = {}
	ordered_importances = {}
	min_split_num = int(minsplit * len(train))
	min_leaf_num = int(minleaf * len(train))

	select_forest_fit = ExtraTreesClassifier(
		n_estimators=100,
		min_samples_split=min_split_num,
		min_samples_leaf=min_leaf_num).fit(X_train, y_train)

	importances = select_forest_fit.feature_importances_
	select_forest_ranks = np.argsort(importances)[::-1]

	```

Tips about these techniques.
* don't assume that all these techniques are helping. Always check if they are helping or hurthing
* don't get to fancy with these techniques. The simpler they are the better. The most straight forward one is the forward selection type of feature selection in which you can see your model selecting features with greater performance scores incrementally.
* You need to setup your model evaluation before you try any of these feature selection methods to see how if any of these are actually helping.
* Try to focus on things built into sklearn because if you are writing custom code, it's easy to make mistakes 
* ML extend is also good as well but just kind of stick with Sklearn
* 

```


## How could you possibly go about selecting the best features for the model?

```
Instinct tells me that I could go with something like the top feature importances from a decision tree, or an ensemble of decision trees (Boostin Gradient Classifiers, Random Forest, AdaBoost)

```


## what is feature engineering and what are some ways that you can create features?

```
Creating features to include in the model so to better model the relationships between the variables and the target.

Feature engineering requires proper feature preprocessing, otherwise, the engineering process may result in bunk data that are not as optimal as we would like to work with. 

Feature processing may be required for each type of data:
1. Numeric
2. Categorical
3. Ordinal
4. datetime
5. coordinates
6. Missing values

# Categorical
Often requires data to be onehot encoded, however, some models like random forest do not require one hot encoding (random forest can split each group within the feature into separate leaves and predict fine probablities). 

The types of Feature preprocessing and feature engineering should be dependent on the underlying algorithm/model.

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
In other words, we introduce a SMALL AMOUNT OF BIAS into how the NEW LINE is fit to the data. But in return for that small amount of BIAS, we can get a significant DROP IN VARIANCE. Ultimately, this reduces the overfitting on the training set while simultateously providing a better fit on the new testing data. In other words, by starting with a slightly worse fit, (L.A.S.S.O.) regression can provide better long term preductions

Regularization helps to do this by introducing a penalty to the model parameters (weights) of the model, and that is multiplied by a lambda coefficient (which determines how severe this penalty is treated), based on some error/cost/loss function.

```

## What is L1-LASSO Regularization/Regression

```
One way to remember LASSO from Ridge regression is that LASSO starts with an L so it goes first in the order of remembering which type of regression is which. It also helps to remmeber that L1-regularization is the type of regularization that takes the absolute value of the penalty term and NOT the squared penalty term.

You can remember that it is Ridge Regression because a ridge is a mountain and you can easily imagine a mountain with two peaks and those are your ridges. You can also remember that Ridge Regression because there are 2 R's and there is the number 2 in L2-Ridge Regression, which is also a pointer to the fact that the penalty term is squared and multiplied by the lambda regularization term (exponent 2).

L.A.S.S.O. Regression is also known as L1 regression or L1 regularization. Typically for zero-ing out the learned feature coefficients for a learning model, meaning that by selecting the features that are not zeroed out, and dropping the use of the zeroed out features in the model, we can eliminate some of the complexity of the model to reduce bias in the model so that we can better learn the relationship between the model parameters, the feature variables and the target variable.

When least squares or linear regression determines what values the parameters of the model it should take, it tries to minimize it's loss/cost/error function:

	sum of squared residuals.

In contrast, L1-L.A.S.S.O. regression tries to minimize it's cost function (the curly braces are just there to make the components of the equation more readable). The slope is the same as the weight/parameter coefficient of feature variable: 
	
	{ sum of squared residuals }  + { lambda * sum(abs(slopes)) }

	AKA 

	{ sum of squared residuals }  + { lambda * sum(abs(learned_feature_weight)) }

	AKA

	{ sum of squared residuals }  + { L.A.S.S.O. Regression Weighted Penalty }
https://www.youtube.com/watch?v=pd-0G0MigUA
# Lambda
Lambda is a regularization coefficient common to both L1-L.A.S.S.O. Regression and L2-Ridge regression. The {slope*2} component adds a PENALTY to the model parameters in traditional least squares cost function and {lambda} determines how severe the penalty is. Lambda can take on any value from 0 to positive infinity. If lambda is 0, then the L.A.S.S.O. regression penalty is also 0. As we increase the the regularization parameter, we increase the effect of the penalty parameter, and thus, the line because less and less sensitive to the effects of the feature inputs. If lambda is infinite, then we are left with a horizontal line, not fitting to any of the data. 

How do we decide what value to give lambda? We just try a bunch of values for lambdaand use cross-validation, typically 10 fold cross validation, to determine which ones result in the LOWEST VARIANCE between training and dev datasets.


# L.A.S.S.O. Regression Penalty
In this example, the {sum of squared residuals} for the *OLS regression line* is 1.69, and for the *L1-L.A.S.S.O. Regression line*, the {sum of squared residuals + L1-L.A.S.S.O. Regression weighted Penalty}  is 0.74. Thus if we wanted to minimize the sum of the squared residuals + the L.A.S.S.O. Regression weighted penalty, we would choose the L1-L.A.S.S.O. Regression line over the OLS regression line. 

WITHOUT the SMALL AMOUNT OF BIAS that the penalty creates, the OLS regression line has a LARGE AMOUNT OF VARIANCE. In contrast, the L1-L.A.S.S.O. Regression Line, which has a SMALL AMOUNT OF BIAS due to the L.A.S.S.O. REGRESSION WEIGHTED PENALTY, has LESS VARIANCE.


Before we talk about lambda, we have to talk about the effect of the l1-L.A.S.S.O. regression penalty has on how the line is fit to the data. When the SLOPE OF THE LINE(learned weight/coefficient of feature variable) IS STEEP, then the PREDICTION FOR THE TARGET VARIABLE IS VERY SENSITIVE to SMALL CHANGES IN THE FEATURE VALUE. By the same measure, when SLOPE IS SMALL, then for every one unit increase in the input feature value, then the prediction for the target variable barely increases.

In the previous example, the L.A.S.S.O. regression line resulted in a line with a smaller slope, which means that predictions made by the L.A.S.S.O. regression line, are less sensitive to the feature values for the 'weight' attribute, than the OLS regression model. 


L.A.S.S.O. Regularization also works when we deal with discrete categorical variables. The slope or feature weight in this case would be equivalent to the difference between the average target attribute value for feature1-group1 vs the average target attribute value for feature1-group2. They can be thought of a binary values. Groups that are assigned the 0 value rely on other terms such as the y-intercept to predict their target value alone. For the Groups that are assigned to the 1 value, they rely on both the y-intercept and the slope/feature weight to determine their target attribute value. 

The whole point of doing L.A.S.S.O. regression is because small sample sizes like these can lead to poor least squares estimates that result in terrible machine learning predictions. 

You can even use L.A.S.S.O. regression for logistic regression problems and the L.A.S.S.O. regression function would look like this instead:

	{sum of squared likelihoods} + {lambda * sum(abs(slope))}

It helps predict whether or not a particular class label is sensitive to the feature variables. When applied to logistic regression, L.A.S.S.O. regression optimizes the sum of the likelihoods as opposed to squared residuals in OLS regression because logistic regression is solved using Maximum Likelihood.

##### L.A.S.S.O. Regression for Regularization in Complicated models #####
Now we've seen how using L.A.S.S.O. regression has been used to reduce variance by shrinking parametersand making our predictions less sensitive to them. We can apply L.A.S.S.O. regression to complicated models as well such as multidimensional or multivariate datasets. In general the L.A.S.S.O. regression penalty contains all the parameters/feature weights except for the y-intercept. 

An equation for a L.A.S.S.O. regression could look like this for a complicated model:

	{sum of squared residuals} + {lambda * sum{abs(slope) + abs(diet_difference) + abs(astrological_effect) + abs(airspeed_scalar)}}

Every parameter except for the y-intercept is scaled by the measurements. 

That said, we need at least two data points for a 1-feature-1-target dataset to even have estimate the parameters of the data. We need at least three data points if we had a 2-feature-1-target dataset.

The rule is then that you require (F+1) data points, where F is the number of variates/features in your dataset.

Having enough datapoints-to-features in a dataset for a large parameter dataset might be bonkers and ridiculous to achieve. But we can solve for this using L.A.S.S.O. REGRESSION. 

It turns out, we can solve for a large feature dataset with less than the required number of samples, using L.A.S.S.O. REGRESSION. L.A.S.S.O. regression can find the solution with cross validation and L.A.S.S.O. regression penalty that favors smaller parameter values.

Summary:
1. when the samples sizes are realtively small , the L.A.S.S.O. regression can improve predictions made from new data(ie. reduce variance) by making the predictions less sensitive to the training data. this is done by adding a L.A.S.S.O. regression penalty to the thing that must be minimized. 

2. lambda is determined using cross validation. 

3. even when there isn't enough data to find the OLS parameterestimates, L.A.S.S.O. regression can still find a solution with cross validation and L.A.S.S.O. regression penalty.

Another note, after having read through L2-Ridge regression, is that L.A.S.S.O. regression tends to do a little better than L2-Ridge regression when most of the variables are useful. When there are lot of useless variables in the model, you get a lot of variance as a result.

4. If there are correlated terms, LASSO tends to pick one of the correlated terms and eliminates the other ones that are correlated. Whereas ridge regression tends to shrink all of the parameters for the correlated variables together.
```

## What is L2-Ridge Regression/Regularization
``````
Read L.A.S.S.O. Regression.

L2-Ridge Regression is SIMILAR to L.A.S.S.O. regression in these manners:

1. Adds a penalty to the parameters/weights/coefficients of the model with the goal of making the model less sensitive to the values training data. This is particularly useful when you have a small amount of training data to work with. Larger coefficients are associated with larger penalties. In this regard, you introduce a small amount of bias into the your data with the goal of reducing the amount of variance that your mdoel has in terms of its fit from the training data vs. the dev/testing data.
2. Ridge also has a lambda parameter which is the regularization parameter. It can be 0 to positive infinity and it determines the strength of effect of the L2-Ridge regularization penalty. Just like L1-L.A.S.S.O. Regression, the lambda for L2-regression can be discovered using cross-validation
3. Just like L1-L.A.S.S.O. regression, they can be applied to complicated models that combine discrete and numeric/continuous data types. 
4. just lihe L1-L.A.S.S.O. regression, L2-Ridge regression contains all of the estimated parameter/learned_weight_coefficients in the calculation of the penalty, except for the y-intercept.
5. When lambda is 0, there is no regularization and no penalization of the model coefficients, as the penalty has been zero'd out by the lamda, thus the model is essentially just minimizing the sum of squared residuals. 
6. The penalty does not affect all the parameters of the model all the same.

	{sum of squared likelihoods} + {lambda * sum(learned_variable_coefficients**2)}

L2-Ridge regression is similar to L.A.S.S.O. regression except for a few important DIFFERENTiations.

1. it is differentiable.
2. you take the squared value of the slope instead of the absolute value of the slope/parameter/learned_feature_weights
3. The big difference between L1-L.A.S.S.O. regression (which uses the abs(coeffs_)) and L2-Ridge regression (which uses the coeffs**2), is that L1-L.A.S.S.O. regression can only shrink the slop asymptotically close to 0, whereas L2-Ridge regression can shrink the slope ALL THE WAY to 0. 
4. For L.A.S.S.O., no matter how small you make lambda, you can't shrink the coefficients of the features down to zero. However, for Ridge regression, you can shrink the learned parameters down to 0. This can be used as a method to do feature selection/ and reduce overfitting at the same time. It is a little bit better than L.A.S.S.O. regression in this regard at reducing the variance in models that contain a lot of useless variables. This makes the final equation easier to interpret. 
5. Ridge regression tends to shrnk all of the paramters of the correlated variables together. This is the opposite of L1-LASSO Regression, where in L1-LASSO regression, we tend to only pick one of the correlated variables (if there are correlations among variables) and then it eliminates the other ones that are highly correlated. 
``````

## What is Elastic Net Regression/Regularization

```
When you are working with a very complex dataset  with lots of features, you don't perform too well with the LASSO regression with compared to L2-Ridge Regression. It is good to choose the correct model to be able to estimate the parameters of the model using either ridge or L1-LASSO regression. But how do you know which one to use?

The good news is that you don't have to choose which one to select for a given dataset. ElasticNet regression is particularly good when there are correlations between parameters. This is because on it's own, LASSO tends to only pick one of the correlated terms, and eliminate the other.

It is simple if you know about L1-LASSO regression and L2-Ridge Regression already because Elastic Net Regression is just the addition/combination of L1-L.A.S.S.O. Regression (sum of squared errors + lambda*slope_of_penalty/coefficient_of_features**2) and L2-Ridge regression (sum of squared residuals + lambda*abs(feature_coeffs))

It starts with the calculation of the sum of squared residuals/least squares like the others:

	{sum of squared differences} + {lambda1*sum(abs(feature_coeffs))} + {lambda2*sum(feature_coeffs**2)}

	AKA

	{sum of squared residuals} + {LASSO regularization penalty} + {Ridge regularization penalty}


Lambdas1 (LASSO) and 2 (RIDGE) are different lambda regularization terms which can also be discovered or fine-tuned using cross validation on different combinations of lambda1 and lambda2 to find the best combinations.

when Lambda1 and lambda2 = 0, it is essentially just the least squares parameter estimates. If lamda2 =0, we just have the lasso regression, and if lambda1=0, then we have ridge regression.

By combining both LASSO and ridge regression equations, the ElasticNet Regression groups and shrinks the parameters associated with correlated vairables and EITHER leaves them in the equation or removes them all at once. It does a better job with dealing with correlated feature parameters in the model than either of the other two subroutines.
In order to know whether your parameters are highly correlated or not, you can perform a covariance matrix or a test of correlation.
```

## What is Boosting and how can it help in tuning a model?

```
The StatQuest video on XGBoost, Gradient Boosting Classifiers, AdaBoost show an example of boosting in action
```
## What is Bagging and how can it help in tuning a model?

```
The StatQuest video on Random Forest show bagging in action
```


## what is covariance? How is it related to correlation

```

```

## What is the difference between a classification problem and a regression problem?

```
A classification problem predicts discrete class lable outcomes, whereas a regression problem produces inferences for continuous valued outcomes.
```

## Pick a machine learning algorithm and explain how it works.

```
############ Adaboost explained #############

In decision trees a tree with one node and two leaves is called a stump. So this is really a forest of stumps, rather than trees. Stumps are not great at making accurate classifications but they are the basis of AdaBoost.

### Number of Splits ###
In random forest, decision trees are grown to a full size. Some trees might be bigger than others, but there is no predetermined maximum depth. In contrast, in a forest of trees made with Adaboost, the treeds are just a node and two leaves.

### Number of Features to Make Classification ###
The main difference between a tree and a stump is that a stump can only use one feature to make a classification decision. Thus stumps are technically weak learners. However, that is how adaboost likes it. One of the reasons why they are so liked. 

### Weighted Voting ###
In a random forest, each tree has an EQUAL vote on the final classification. In contrast, a forest of stumps created using adaboost, some stumps have more say than others in the final classification.

### Order of creation ###
In a random forest, each decision tree is made independently of the others. In Adaboost, the order in which the trees are made matter. They are sequential classifiers, meaning that the errors the previous stump makes informs how the subsequent stump is created.

Main Ideas:
1. adaboost combinses a lot of weak learners to make a classification. The weak learners are almost always stumps. 
2. There is weighted voting
3. Sequential learning from errors made from the previous stump

### Creating AdaBoost from scratch ###

When you start off, each SAMPLE or DATA POINT is assigned a weight attribute. At the start, all samples get the sample weight:

	1/Total Num of Samples

When we start, all samples have the same weight, however after we make the first stump, these weights will change, how the next stump is created. These sample weights will be updated. 

We make the first stump in the forest after initializing the equal sample weights. We select a variable that does the best job at classifying the samples. Because all the weights are the same, we can ignore them right now. First we go through each feature individually, and see how many those feature values help us classify the labels. Note that for numeric features, we have to determine the best value to split on, based on some techniques in the Decision Tree StatQuest Video. 

Now we calculate the Gini Index for the three initial stumps created from the three initial features. The Gini index with the lowest value is the one that does the best job at classifying, so this will be used in the creation of the first stump in the forest. Now we will have to determine how much say this stump has in the classification. We determine this weight by how well it classified the samples. 

The TOTAL ERROR for the stump is the sum of weights associated with the INCORRECTLY CLASSIFIED SAMPLES. Initially this weight is 1/TotalNumSamples, So because all the total weights add up to 1, total error will always be 0 for a perfect classification and 1 for a horrible stump. 

We use the total error to determine the weight of this first stump in the final classification with the following formula (note that if total error is 1 or 0, this weight equation will freak out so we add a small error term from making this happen):

	Vote_Weight = 0.5*log(1-total_error/total_error)

When a stump does a good job and the amount of error is small, then the Vote_Weight, is a relatively LARGE POSITIVE VALUE. When a stump is no better at classification than a flip of a coin and total error is 0.5, then the vote_weight=0. When a stump does a terrible job, and the total error is close to 1, in other wrods, if the stump consistently gives you the oppositive classification, then the amount of say will be a large negative value. 
######## End of the first pass ##########

######## Start of the Second Stump #######
######## Updating the Sample Weights #####
Now we need to learn how to update the sample_weights so that the next stump will take those sample errors that the current uses to classify and calculate it's vote_weight.

When we created the first stump, all of the sample weights were all the same, meaning that we did not emphasize the importance of correctly classifying any particular sample. But since the first stump incorrectly classified some particular samples, we will emphasize the need to classify it correctly by increasing it's sample_weight and decreasing all of the other sample_weights. 

Let's start by updating the sample weights. 

This is the formula by weight we update the sample weights for the INCORRECTLY CLASSIFIED SAMPLES:

	updated_sample_weight = sample_weight*e**vote_weight

	example:

	updated_sample_weight = {1/TotalNumSamples}*e**{first_calculated_vote_weight}

to get a better understanding to voting weight has an effect on euler's number in the calculation of the updated_sample_weight, we need to draw a graph. When the previous classifier has a big vote_weight because it did a good job classifying, then we will scale the sample_weight with a large number, that means that the new sample weight is much larger than the old one. And when the vote_weight is small because the previous weak learner did a bad job at classifying, then the previous sample weight is scaled by a relatively small number. That means that the new sample weights will only be a little larger than the old ones. 

This is the formula for updating the correctly classified samples (note that the vote_weight is now negative in this formula. this is the main distinction between the two):

	updated_sample_weight = sample_weight*e** -vote_weight

Just like before, we will have to plot it to see the effect of raising e to a negative vote_weight. When the amount of say is relatively large, Then we scale the sample weight by a value close to 0. This will make the new sample weight very small. If the amount of say for the last stump is relatively small, then we will scale the sample weight to a value close to 1, this means that the updated_sample_weight will only be a little smaller than the old one. 

We will keep track of the new sample weights in this new column. After we have calculted all the sample weights, we will need to normalize the sample weights so that they add up to 1. To normalize means that you make the sum of all the values in a collection of values =1) Currently in the example, if you add up all the next sample weights you will get a value that is less than 1. To normalize:

	Normalized_updated_sample_weights = updated_sample_weight/Sum_Total_Updated_Sample_weights

These are the values we will use for the next stump.

### Weighted GINI FUNCTIOn ###
Now we can use the modified sample weights to make the next stump. In theory, we can use the sample_weights to calculated the Weighted GINI Indexes to determine which variable to split the next stump. The weighted GINI index would put more emphassis on correctly classifying the samples with the larger weights (wrongly classified samples from the previous iteration). 

#### ALTERNATIVE APPROACH instead of using the Weighted GINI function ####
Alternatively, instead of using a weighted GINI index, we can make a new collection of samples that contains duplicate copies of the samples with the LARGEST SAMPLE WEIGHTS. We start by making a new but empty, dataset that is the same size as the original then we pick a random number between 0 and 1. Then we see where that random number falls when we use the sample_weights like a distribution. 

We sample from our original dataset into our new empty one WITHOUT REPLACEMENT randomly. Now, we give each of the samples in the new collection EQUAL SAMPLE WEIGHTS, JUST LIKE BEFORE. However, this doesn't mean that the newxt stip will not emphasize the need to correctly classify these samples. Because some of the samples may have been sampled multiple times and are in essence the same sample that was in the original dataset*N times, they will be treated as a block, creating a large penalty for being misclassified.  

Now we go back to the stump that does the best on classifying the new collection of samples. That is how the errors of the first tree makes influences how the second tree is made and so on and so forth. 

A forest of stumps place their weighted votes together. We add up all the vote_weights for each label or category for the class label, and then pick whichever has the higher value, gets say in teh final predicted label for this record/sample/datapoint. 

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
## What Techniques are there for feature engineering/feature selection/dimensionality reduction?

```

```
## What is PCA? How is it performed and what does it do?

```
Allows us to have a deeper insight into your data visually.

PCA is done step by step with SVD (singular value decomposition). PCA takes high-dimensional/high-feature data, and tells us which genes or variables are most valuable for clustering our data. 

# Tips
1. scale each data point by the standard deviation for that feature
2. Center your data
3. How many principal components can you expect to find? Technically, there is a PC for each feature in the dataset. However, if there are fewr samples than variables, then teh number of samples puts an upper bound on the number of PCs with eigenvalues greater than 0. 

	max_num_PCs = min(len(df.columns[df.columns != "target"]), df.shape[0])

Steps
1. plot data (feature 1 along the x-axis and feature 2 along the y-axis)
2. take averages values of each feature
3. shift the data so that it is centered about the origin of the graph (standardize values). Note that the positions on the graph do not change in terms or orientation, but only a geometric, diagonal shift.
4. We will be focusing on the graph so we will no longer need the rodinal data
5. draw a random line through the origin, and keep doing it until it fits the data. PCA decides whether this line is a good fit or not. 
6. PCA decides whether this random line is a good fit or not by projecting the data onto the line. The distance of the point from the origin doesn't change when new random lines are generated but when we project the point onto the line, we get a right angle between the randomly generated line and the plotted point itself. This makes use of the pythagorean theorem to show how the long and short are related to the line and origin. 
7. We intuitively then, want to minimize the short of the triangle of the lines that create the right triangle for the point of interest, the origin and the line. and then it measures the distance from the data to the line and tries to find the line that minimizes those distances OR it can try to find the line that maximizes the distances from the projected points to the origin. 
8. PCA finds the best fitting line (through trial and error) by MAXIMIZING the sum of squared distances from the PROJECTED POINTS to the origin. We square them because negative values don't cancel out positive values. This best line because principal component 1 or PC1 for short. PC1 has a slope as well, just like any other line and a y-intercept of 0. That means for every point that we go out along the {feature 1} axis, we got up one unit along the {feature_2} axis. This could be interpreted as, to make PC1, you need one part {feature_1} to 1 part {feature_2}. The ratio of feature_1 to feature_2 tells you how much more important one feature is than the other when it comes to describing how the data are spread out. Mathematicians call this a "LINEAR COMBINATION" of feature_1 and feature_2. That is to say that PC1 is a "linear combination of variables feature_1 and feature_2"; they are talking about the slope ratio of features. 
9. We can then solve for the pythagorean with the ol:
	a**2 = b**2 + c**2

When you do PCa with SVD, the recipe for PC1 is scaled so that the length of of the short=1 (c). This is equivalent to by dividing each measurement by the standard deviation for that feature. 
	
	std dev = math.sqrt(sum([(x-mean(X))**2 for x in X])/(len(X)-1))

We can scale all the of the sides of the triangle by the hypotenuse that is (math.sqrt(a**2)) so that everything has a unit scale of on. The new values to the slope of this PC1 line, changes our linear combination recipe values but the ratio of rise over run is still the same. 

This vector that belongs to PC1, which contains the rise and run (feature_2/feature_1), is called the "Singular Vector" or "Eigen Vector" and the proportions of each feature are called "Loading Scores". PCA calls the sum of squared distances for the best fit line, the "Eigen Values" for PC1, and the Square Root for the "Eigen Value for PC1" is called the "Singular Value/Eigen Vector" for PC1. 

	Sum of Squared Distances for PC1 = Eigen Value for PC1

	math.sqrt(EigenValue for PC1) = Singular Value/ Eigen Vector for PC1

	math.sqrt(Sum of Squared Distances for PC1) = Singular Value for PC1


#### Principal Component 2 ####
Now, let's work on the second principal component.
Because this plot in the example is only two dimensional, PC2 is simply the line through the origin of the plot that is perpendicular to PC1, without any further optimization that has to be done. 

This means that the Eigenvector contains loading scores for negative(feature_1) and feature_2 so (feature_2/negative(feature_1)). If we scaled everything so that we get a unit vector (i.e., scale the values by the value of the hypotenuse of the right triangle), then we get the eigenvector for PC2, aka the singular vector for PC2. The loading scores for PC2 are the ratio values that each feature contributes to the slope of PC2 (i.e., the proportion is one loading score, and the run value is the other loading score). Again, the loading scores tell us that, in terms of how the value are projected onto PC2, bigger value is X times more important than smaller value.

Lastly the eigenvalue for PC2 is the sum of squares of the distances between the projected points and the origin.

#### Putting It all Together ####

To put it all together and plot it on the plot, we simply rotate everything so that PC1 is horizontal and then we use the projected points to find where the samples go on the PCA plot. For example, if we have a 2 dimensional plot, then each sample has a pair of corresponding values, corresponding to values along PC1 and PC2. That's how PCA is done using SVD (singular value decomposition). 

	Eigen value for PC1 = Sum of Squared Distances along PC1

	Eigen value for PC2 = Sum of Squared Distances along PC2

Remember, we got the eigenvalues by 1) projecting the data onto the principal components; 2) measuring the distances to the origin 3) squaring each distance value 4) adding them all up 5) we can convert them into variation around the origin (0, 0) by diving by the sample size minus 1 (n-1).

	Variance/variation for PC1 = Eivenvalue PC1/(n-1)

	Variance/variation for PC2 = Eivenvalue PC2/(n-1)

	Total Variance around both Principal components = Variance(PC1)+Variance(PC2)

	That means that var(PC1)/totalvar * 100 = PC1 accounts for {}% of the total variation around the principal components.

TERMINOLOGY ALERT: 

Scree Plot: is a graphical representation of the percentage of variation that each Principal component accounts for in the calculation of the total variation around the principal components. It is just a bar plot of the percentages.

#### More: Principal Component 3 ####

PCA with 3 features is pretty much the same as 2 using 2 variables. 

PCA steps:
1. Centre the data around the origin
2. Perform SVD to get PC1, the best fitting line that maximizes the sum of squared distances from the origin, given the projected data points. In the case of having three features/dimensions, you have 3 loading scores for this linear combination of features - one for each feature. The largest of the loading score ratio values indicates that it is the most important feature/ingredient for PC1. 
3. You then find PC2, the next best fitting line that goes through the origin that is perpendicular to PC1. The loading scores for PC2 will be different from PC1 and some features in this principal component maybe considered more important than others this time around. 
4. Lastly, we find PC3, the best fitting line that goes through the origin and is perpendicular to both PC1 and PC2. 
5. If we had more features, we'd just keep on finding more and more principal components by adding perpendicular lines and rotating them. In theory, there is one per feature but in pracitce, the number of principal components is either the number of features or the number of samples, whichever is smaller. 
6. Once you have all the principal components figured out, you can use the eigen values (sum of squared distances from the origina for each principal component) to determine the proportion of variation that each PC accounts for...You do this by first summing all the variation that each component contributes (again, that is eigen value/num_samples-1).
7. Once you have the scree plot up of the PCs, you can get a good approximation of the higher dimensional graph and can tell where about 94% of the variation in the data come from in terms of principal components.
8. To convert the 3-D graph to a 2-D PCA graph:
	i. we just strip away everything but the data and PC1 and PC2
	ii. project the samples onto PC1
	iii. and PC2
	iv. rotate until PC1 is horizontal and PC2 is vertial (this just makes it easier to look at).
	v. we look at the project points for each sample, and plot them where the projected points coordinates for the corresonponding sample would intersect on the plot.

We can still draw a 2-D PCA graph with the top 2 principal components or 3-D graph if we pick the top 3 principal components (if we have enough principal components to work with. Remember, the number of components is the minimum of either the number of features or the number of samples, whichever is lower).

even noisy PCA plots where the PCs have high contribution can tell a story about how the data are cluster together.

```

############################################### Statistics #################################################################

## Explain Bayes Theorem

```
Equation for Bayes Theorem

	P(A|B) = (P(B|A)*P(A))/P(B)

	OR

	P(B|A) = (P(A|B)*P(B))/P(A)


```

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
## what is the difference between a probability distribution function and a probability density function?

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
We us a KDE function When we try to estimate the underly probability density function of the underlying data.

Intuitively: What % of the dat falls between a particular definite integral (area range region underneath a probability density curve).

The kernel density estimator/function has some of the same intuition has being able to get the area underneath the curve at particular definite integrals underneath the curve. It is similar to the histogram function, in that it is an averaging function of the width of the bins of a K(x) function (K is a Kernel weight function). Kernels weigh observations differently depending on how far away they are from the point X, the point where we are evaluating at f(x) and it does this over all of the i observations. There are lots of kernels, but the Gaussian or the PDF (probability density function) of the normal distribution is one. You can have a Kernel that is an indicator function if you are working with discrete intervals or bins, but if you are working with something like a Gaussian kernel, it applies a continuous weight that is decreasing the further we move away from point X.

For a KDE function, you have to choose b, which is the bandwidth (think the width of the bin for a histogram bar except this is a continuous value for the probabiltiy density distribution). 


```

## what are the Kolmogorov Axioms

```
Kolmogorov axioms are a fundamental part of probability theory. In it, the probability P of some event E, denoted P(E) is usally defined to satisfy these axioms. The axoims are described below. These assumptions can be summarised as follows: 

Axiom #1:
The probability of an event is a non-negative real number

Axiom #2:
This is the assumption of unit measure: that the probability that at least one of the elementary events in the entire sample space will occur is 1. The set of all possible outcomes P(S)=1, where S is the sample space of an experiment. The probabilty of any of the outcomes happening is 100%

Axiom #3:
If A and B are mutually exclusive outcomes, P(A u B) = P(A) + P(B). Here u stands for the 'union' operator. We can read this by saiying that if A and be are mutally exclusive outcomes, then the probabiltiy of either A or B happening is the probability of A happening + the probabilty of B happening.

```

## what is multicollinearity. Why is it bad? How do you deal with it? How do you test for it?

```

```

## what is homoscedasticity. Why is it bad? How do you deal with it? How do you test for it?

```

```

## What is the Students t-test and when would you use it?

```

```

## P-Values


```

```

## Assumptions of Linear Regression

```

```

## Parametric vs Non-Parametric

```

```


## How do you test for normality of distribution?

```

```

## What is ANOVA and when do you use it?

```

```

## Explain Bayes Theorem

```

```

## What is Maximum Likelihood?

```
The goal of maximum Likelihood is to find the optimal way to fit a distribution to the data. There are lots of types of distributions for different types of data. There are normal distributions, poisson distributions, gamma distributions, exponential distributions...

The reason you want to fit a distribution to your data is that it can be easier tow ork with and it is also more general - it applies to every experiment of the same type. 

In this case, we believe that the feature values might be normally distributed before hand (looking at it specifically). 

Once we have an idea of what the distribution might look like, we have to centre the thing. Is one location better than another? 

A normal distribution says:
"Most of the values you measure should be near my average/centre of the hill". 

We could then shift the normal distribution so that the mean of the normal distribution is the same as the average feature measurement? 

# Plotting/Discovering the Maximum Likelihood
We can plot the likelihood of observing the data against the location of the center of the fitted distribution (likelihood on the y-axis and feature values on the x-axis). We start on the left side of the plot, and we calculate likelihood of observing the data, and then we shift the distribution to the right of the data points lined-up data points (along a number line) and recalculate. We just do this all the way down the data. Once we tried all the possible locations we can center the normal distribution on, We want the location on the plot that maximizes the likelihood of observing the weights we measured(i.e., x-coordinate value that maximizes the y-axis value, since the y-axis value is the likelihood). 

Thus, this point becomes the "maximum likelihood estimate" for this mean (specifically in this case, we are talking about the MEAN of the DISTRIBUTION, and NOT the data) - however, with a normal distribution, those two things are the same. Good - now we have the maximum likelihood estimate for the mean. 

NOW, we have to figure out the maximum likelihood estimate for STANDARD DEVIATION OF THE DISTRIBUTION. Again, we can plot the likelihood (y-axis) of observing the data against standard deviation values (x-axis). Now that we have found the standard deviation that maximizes the likelihood of observing the weights we measured. 

This is the normal distribution that has been fit to the data by using the maximum likelihood estimates for the mean and standard deviation (of the distribution). 

Conclusion:
Now, when someone says that they have the maxmimum likelihood estimates for the mean and standard deviation, or for something else, you know that they found the value for the mean or the standard deviation (or for whatever) that maximizes the likelihood that you observed the things you observed. 


TERMINOLOGY ALERT:
In everyday conversation, "probability" and "likelihood" mean the same thing. However, in stats-land, "likelihood" specifically refers to this situation we've covered here; where you are trying to find the optimal value for the mean or standard deviation for a distribution given a bunch of observed measurements. 

The equation for the normal distribution is (you just have to plug and chug and you get the likelihood value along the y-axis given a value for mean (u) and standard deviation (sigma), given a value of x in a collection X):

	pr(x|u, sigma) = (1/math.sqrt(2*math.pi*vsigma**2)) * math.e**-((x-u)**2/(2*sigma**2))

To solve for the maximum likelihood estimate for u, we treat sigma like a constant and then find where the slope of its likelihood function = 0. We do the reverse if we want to solve for sigma instead of u. 

We solve this for all data points and the MLE for more than one data point is just

	MLE of mean for all data points = 
	MLE(x1) * MLE(x2) * ...MLE(x[len(X)])

```

## Difference between likelihood and probability

```
# Probabilities 

Are the AREAS under a FIXED distribution

	pr(data | distribution)

Probability = chance that a randomly select sample drawn from the distribution will have a particular value. With probability, we keep the right hand side the same (i.e., the shape and location of the distribution) while we change the measurements on the left side of the expression.

Notation:

	pr(mouse_weights=32-34 | mean=32, stddev=2.5) = 0.29

# Likelihoods

Are the y-axis values for FIXED DATA POINTS and distributions that CAN BE MOVED. 
	
	L(distribution | data)

Likelihood = the likelihood of a distribution with a mean=32, and a stddev=2.5, given that we have already measured and weighed a 34 gram mouse). With likelihoods, the measurements on the right side are fixed, whereas the the shape (stddev) and location(mean) of the distribution with the left side.

Notation:

	L(mean=32, stddev=2.5| mouse_weights=34) = 0.12
```

## neural networks

```

```

## Git


## Docker

## AWS