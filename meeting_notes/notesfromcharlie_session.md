notes from the session with Charlie:

one hot encoding
knn imputation

class imbalance:
change the class weights
penalization for class imbalance problem
set it to the relative ratio of your class imbalance
* set hypermparameter of the loss function based on this ratio
* set his lambda to .7 for the rare class and .3 tfor the very rare class

if you are doing a regression problem:
* if you have outliers, if your outliers don't make any sense, you can drop them

# image annotations working with medical image datasets
center for clinical datascience

massachusetts partner healthcare institutions
clinical innovation fellows (PMs who have MDs). liason that goes between data science team and project partners between what they want and expect. for time frame and scope of the project

They help with data annotations. Deliverables really depend on the project and what the project partners want and what the resources that are allocated tot he project. 

it depends on the type of project for the tyhe types of deliverables.

codify the deliverables into specs. what the model should do, what it shouldnt do, performance metrics.

half research centered, half production of machine learning models in healthcare. working with industry partners with NVIDIA, GE. build AI models and trian them on the data. HAnd them off to them to get FDA models. 

Reports are all outlined in the Statement of work. There are milestones that you are aiming for when you work with project partners for these timelines/deadlines. This is the back and forth between the project partners, the engineering team and data team.

on-boarding process is super important (mentoring) to get up to speed

year and half of interviewing.

Back to visualizing your data

stationarity: the distribution is changing over time. You want to filter out the noise. You do some tests to see if your time seriers data have stationarity:
* Dicky Fuller

make your data stationary by differencing it. It filters out the trends. kind of like a normalizing technique. makes things on the same scale. you can even try to smooth it like with exponential smoothing to reuduce the variance of the noise. 

line plots for time seriers data. 

make sure your plots are appropriate for the data type/nature that you are working with. 

image data:

* you can do a histogram of the pixel values
* pixel density values and distribution to see if they are relative similar or if there are outliers
* edge filtering
* reduce salt and pepper noise
* normalize (- mean / difference) -> white balance normalization...or mean whitening...which is a preprocessing step.
* you can do this image by image; across entire batch of data; or by rgb channels
* resizing but make sure that the ratio ios the same. You want to keep the semantic image the same
* if the transformation is affine, meanign that it preserves lines, then that is fine. 
* Usually you don't want the sizing to be too extreme. Be similar ration or a square


# Industry, there is nothign too trivial

Deliver value as quickly as possivble. you might not have to deal with the most SOTA, PCA linear progression , decision trees, KNN, is yoing to be your bread and butter most of the time.

As a data analyst it encompasses everything. data analyst and business analyst.
* an undervalued skill in data science: communicating the results of wwhwatver you are doing to a project stakeholder to the project manager a

* figuring out the right level of detail to divulge to your audience.
* figuring out the context
	* mainly care about the results. 
	* what is pertinent to the immediate case.

* get really good at writing concisely and too the point
* Most of your time is making slides and presenting. 
* making visualizations
* sending out emails to get the information you need to clarify requirements

* get really good at one visualization tool. know how to do all the basic stuff in that. This holds true for a lot of things. 
* better to be specialist than a generalist because that gives you an edge.

# soft deliverables: 
- dirty data set cleaning
- feature engineering and selection for that dataset