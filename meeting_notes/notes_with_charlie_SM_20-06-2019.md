# Meeting Minutes with Charlie June 20, 2019


## Talking point and questions
* Should I abandon that idea of webscraping and select a different set of data to scrape from that has more certainty or should I just go with the webscraping dataset from Entomo Farms
* I don't have a clear vision as to what I can or what I want to do with natural language processing with the Entomo Farms recipe dataset
	* Visualization with wordcloud
	* wordembeddings
	* What is the value that can be had from doing natural language processing from a recipe or cook book. I'm kind of lost here as to what this can do
	* What I know from NLP: sentiment analysis (for social media text such as tweets and reviews), machine translation, captioning, chatbots, answering questions
	* It takes some creativity and thinking as to what one can or what one wants to do with a data after one has it


## Soft Deliverables:
* get a review of the resume. put it in the slack channel. for jeremie and for other peers to review
* in a couopel of weeks you can start applying
* then from there you can start AB tesing your resume
* take the data that 
* slim down your resume
	* limit experience to 3 bullet points
	* get rid of the references
	* leave off certificate dates
	* leave off some of the courses
		* leave of project management?
* be more concise and focused on the skills that I have to offer instead of just going for volume
* be more concise with no as many words for the experience section
* Don't try and dilute what is impressive about what you've done
* the signal to noise ratio is worst
* in skills:
	* github
	* git, version control, docker, keras, tensorflow
	* instead of SageMaker, just use AWS
	* leave out Jupyter Notebook. Just what language you used is fine
	* maybe no RedCap


## The Project
* Explore the dataset. Think about how I want to tackle the problem
* HOw do you deal with things like missing data, large datasets, think about feature selection from 300 different features
* piece meal, explore, EDA, make some plots. Think about class imbalances in the dataset
* make a bunch of different plots. SEe if the performance varies by different groups with the .groupby() function in pandas
* does it vary by income level. look for correlations and collinearity of features
* removing duplicitous features
* final project:
	* evaluate against the test set to see how well the model against the test set
	* a blog post of why I chose something against another
	* wrap it up into a simple application, where a user can inpute their application and then serve them whatever rpeciction and see how well they would perform at their job
* Have a notebook done about what you did, why you went about it this way, any blockers, along the way.
* a script or a notebook about what you did.
* How you dealt with the missing data and correlations.
* after a certain amount of time, you can and should reach out with Charlie for help, rather than just spending more time on it.
* learn Dask


## Takeaways
* Half the job is figuring out the requireemnts, pain points and questions 
* Commjnication is an underated skill 
* Maybe new dataset from Charlie to work with
	* What predictors
	* What performance of each person on these roles would be. 
* he would derive a lot of benefit from the questions and interview portion
* I prefer an end to end
* How to go about asking for help and WHEN:
	* document what you tried and what wasnt working
		* search terms, links and methods
	* how you thought the problem
	* try reading google or stack exchange
	* don't tell them why you might think is the issue. Tell them the error messages
		* sometimes the problem isn't what you think it is
	* usually there are diminishing returns on how long to spend on something....
	* if you spend too long, it is not as good, since you are wasting time that can be spent on better things
	* amount of time you spend should be proportional as to why something isn't working.
	* it's good to have people there to give you esimats on how long something should take.