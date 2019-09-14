

print(df.describe())

3. EDA - Missing Values
We print out the the counts of observations with missing values in this dataset

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

​

# There are a lot of missing values in this dataset

print(df.isnull().sum().sort_values(ascending=False)[df.isnull().sum() > 1000][:25])

Dropping Redundant Features

Based on the codebook and the naming conventions for this dataset, we will explore and determine which features could be dropped due to overlapping measures. Sometimes this is done as a result of differences in coding schemes or categorical aggregations of the result. We will do some investigation on these features, their level of correlation and multicollinearity with the target variable and decide which features we can drop during the preprocessing step.

# look at these features individually and drop stepwise after analysis

​


Preprocessing (Dropping and Replacing) for Missing/Unavailable Value Encodings

With this particular dataset, some missing or unavailable column values are encoded as a numeric value. Some particular values that were used to encode these values are: 999, 9996, 9997, 9998, 9999, 99999. Let's have a look at them first. Our next step would be to map these values to np.nan for imputation afterwards. We make a list of these notable features.

Also note that for some features, the integer values or float are actually strings, thus making the feature column itself an object column. What we would want here is to cast these specifically as integers, after having dealt with the letter encodings. Using Int64

    v71 -> cast letters to np.nan, cast values to np.nan, cast all to Int64
    v105 -> drop

Letter columns that use numeric encodings to indicate missing:

    isic1c -> replace num string as np.nan

possible_missing_data_codes = ['999', 9995, '9995', 9996, '9996' ,9997, 9998,'9998', 9999, '9999', '99999']

Drop v105 because it is the same as isic features and we already drop those.

We then impute the missing/unavailable encoded values in isic1c with np.nan

# v105 is to v71 what isic1l is to isic1c; one refers to industry of last job and one refers to industry of current job

df.drop(['v71', 'v105'], inplace=True, axis=1)

df.isic1c.unique()

missing_values_to_replace = ['9995', '9996', '9997', '9998', '9999']

​

df['isic1c'].replace(to_replace=missing_values_to_replace, value=np.nan, inplace=True)

Drop Missing Data by Axes
Missing Data Proportion by Columns

We will want to drop a number of the columns have have a significant amount of its data missing. Some features have disproportionately more data missing than others and there will be no way to impute those features. In which case dropping those features might be the best course of action to clean up the dataset. Feature column values that have a proportion of their data missing equal to or greater than an arbitrary threshold will have their columns dropped. Afterwards, a row-wise operation of this procedure will be further applied to include cleaner samples in the dataset.

We will examine to see which features reach this threshold requirement to be dropped before we do any such dropping.

Afterwards, we will want to smarly impute the remaining missing data points as much as we can.

# features by % of missing data

dropthreshold = 50

​

pct_missing_list = list([col for col in df.columns if df[col].isnull().sum()/df.shape[0] * 100 >= dropthreshold])

pm_list = []

​

​

for col in df.columns:

    if df[col].isnull().sum()/df.shape[0] * 100 >= dropthreshold:

        pm_list.append(df[col].isnull().sum()/df.shape[0] * 100)

        

dropped_columns_dataframe = pd.DataFrame(pm_list, columns=['% Missing Data'], index=pct_missing_list).sort_values(by='% Missing Data', ascending=False)

​

# get the data points in the dataframe that have >= 50% of their data missing

df_col_drop_gtoe_50 = df[pct_missing_list]

kept_feat_cols = list(set(df.columns) - set(pct_missing_list))

reduced_df = df[kept_feat_cols]

​

print("The features with {}% or more of it's data missing will be dropped from the dataset.".format(dropthreshold))

print("{} Columns will be dropped, which constitutes a {}% drop in the number of features from the original dataset".format(

    len(df.columns) - len(kept_feat_cols), 

    ((len(df.columns) - len(kept_feat_cols)) / len(df.columns)) * 100

)

     )

​

# save the results to csv to look at later

dropped_columns_dataframe.to_csv('./features_greater_50_percent_missing_data_by_cols.csv')

​

# write out the dataframe after having dropped the columns

reduced_df.to_csv('./reduced_df_gtoe_{}_missing_df.csv'.format(dropthreshold))



def report_missing(preproc_data, postproc_data, axis=1, dropthreshold=None):

    """ dataset in dataframe format passed in is before and after dropping feature columns or rows. """

    print("The features with {}% or more of it's data missing will be dropped from the dataset.".format(dropthreshold))

    

    if axis == 1:

        print("{} Columns will be dropped, which constitutes a {}% drop in the number of features from the original dataset".format(

        len(preproc_data.columns) - len(postproc_data.columns), 

        ((len(preproc_data.columns) - len(postproc_data.columns)) / len(preproc_data.columns)) * 100

    )

         )

    

    else:

        print("{} Data Points will be dropped, which constitutes a {}% drop in the number of data points from the original dataset".format(

        preproc_data.shape[0] - postproc_data.shape[0], 

        ((preproc_data.shape[0] - postproc_data.shape[0]) / preproc_data.shape[0]) * 100

    )

         )


# Data points with percentage of data missing and greater will be dropped from the dataset

dropthreshold = 20

​

# get the data points in the dataframe that have >= 30% of their data missing

to_drop_df = reduced_df.loc[(reduced_df.isnull().sum(axis=1)/reduced_df.shape[1]*100) >= dropthreshold]

row_reduced_df = reduced_df.loc[(

    reduced_df.isnull().sum(axis=1) / reduced_df.shape[1]*100 < dropthreshold

)]

​

report_missing(reduced_df, row_reduced_df, axis=0, dropthreshold=dropthreshold)

​

# give the new dataframe a better name

final_kept_df = row_reduced_df

​

# save the dataframe to disk after the processing

final_kept_df.to_csv('./final_kept_df.csv')

Missing Data by Data Type

We can see that there are a considerable amount of missing values for this dataset. We will have to dig in to understand which features would have missing values, and can we come up with a strategy to handle it.

# separate out the numeric and categorical variables to see how much of each are missing

def df_by_type_splitter(dataframe):

    """ a larger dataframe into immediately identifiable numeric and other type dataframes"""

    num_df = dataframe._get_numeric_data().copy()

    cat_df = dataframe.select_dtypes(exclude = [int, float]).copy()

​

    return num_df, cat_df

​

​

numeric_df, categorical_df = df_by_type_splitter(row_reduced_df)

​

​

print("Number of Numeric Features: {}".format(numeric_df.shape[1]))

print("Number of Categorical Features: {}".format(categorical_df.shape[1]))

Numeric Feature Missing Values

It still appears that after dropping the data points and features with a majority of their data points missing, that about a third of the numeric features containg NaN/missing values in more than 20% of their data points.

# take a look at the missing values from the numeric features dataframe

print("Proportionally, numeric features contribute {0:.2f}% of the total features in the dataset".format(numeric_df.shape[1]/final_kept_df.shape[1]*100))

a = pd.Series(numeric_df.isnull().sum().sort_values(ascending=False)/(numeric_df.shape[0]) * 100)

print(a[:12])

a.plot(kind='bar')

​

# 9 numeric features of 37 with more than 20% of it's data missing

print("The number of numeric features with more than 20% of its values missing is: {}, which is {}% of all numeric features".format(9, (9/numeric_df.shape[1]*100)))

We are going to explore correlation/collinearity between the numeric features and drop redundant features.

# categorical feature with numeric encoding

​

cat_feat_with_num_encoding = ['v224','v239', 'isco2c']

for i in cat_feat_with_num_encoding:

    print("This categorical feature with numeric encodings has {} unique values: {}".format(i, len(numeric_df[i].unique())))

Although semantically inappropriate, let's, as an exercise, treat these high-category numerically encoded features as numeric features. Depening on the results of our exploration, we may go back to treating these features as categorical variables in the final model.
Pre-Imputer Imputation for numeric features

Some further processing can be done before the imputation step such as (dropping,casting, merging, etc). The next few segments of code will go through the next preprocessing steps as outlined above.

# we drop 'age_r' as there is another categorical feature containing the same demographic information with more completeness, with the tradeoff of numeric granularity

​

numeric_df.drop(columns='age_r', inplace=True, axis=1)

# we drop 'row' because this is a pretty useless category

​

numeric_df.drop(columns='row', inplace=True, axis=1)

Because v239 has many numeric encodings as a categorical feature, we can initially treat it as a numeric feature and then see whether there is a relationship between it and the response variable: 'job_performance'. We keep the variable if it is a strong predictor of employee job performance; otherwise, we drop the largely-categorized variable.

As we can gleen from the heatmap below in Seaborn, there is no correlation between the feature v239 (current occupation) and job performance of an employee, if current occuptational coding were treated as a numeric variable.

import seaborn as sns

​

o = pd.DataFrame(numeric_df['v239'].fillna(value=numeric_df['v239'].median()))

o['job_performance'] = numeric_df['job_performance']

corr = o.corr()

sns.heatmap(corr, annot=True)

What happens when we look at v239 as a categorical feature? Well, we don't really get to see much of anything at all from onehot encoding this highly categorized feature.

get_dummy_features = pd.get_dummies(o['v239'], drop_first=True)

temp_df_dummies = pd.concat([o[['job_performance']], get_dummy_features], axis=1).iloc[4:10]

print(temp_df_dummies.head())

​

corr = temp_df_dummies.corr()

sns.heatmap(corr, annot=True)

So it looks like for now, it is safe to drop this feature

# Dropping feature 'v239'

​

numeric_df.drop('v239', axis=1, inplace=True)

We will then choose to impute the rest of the missing numeric features, during our imputation step.
Examination of the categorical features

# take a look at the missing values for categorical features dataframe

print("Proportionally, categorical features contribute {0:.2f}% of the total features in the dataset".format(categorical_df.shape[1]/final_kept_df.shape[1]*100))

print((categorical_df.isnull().sum().sort_values(ascending=False)/(categorical_df.shape[0]) * 100 )[:20])

b = pd.Series(categorical_df.isnull().sum().sort_values(ascending=False)/(categorical_df.shape[0]) * 100 )

b.plot(kind='bar')

​

# 17 features with more than 20% of its values missing of 205 features

print("The number of categorical features with more than 20% of its values missing is: {}, which is {}% of all categorical features".format(17, 17/categorical_df.shape[1]*100))

temp = pd.DataFrame(categorical_df.columns, columns=['feature_name'])

temp['num_unique_values'] = [len(categorical_df[col].unique()) for col in categorical_df.columns]

print("There are {} features that have <=10 unique values in feature column, which could benefit greatly from categorization.".format(temp[temp['num_unique_values'] <= 10].sort_values('num_unique_values', ascending=False).shape[0]))

print("This is {0:.2f}% of the categorical features that could benefit from categorization optimization".format(temp[temp['num_unique_values'] <= 10].shape[0]/categorical_df.shape[1] * 100))

# we can see that the categorical features, because of many data points and features there are, contribute heavily to the memory usage

​

categorical_df.info(memory_usage='deep')

# have a look at the memory usage of the top 10 object columns before categorization

​

categorical_df[[col for col in categorical_df.columns if len(categorical_df[col].unique()) <= 10]].memory_usage(deep=True)[:10].sort_values(ascending=False)

We can cast some of the object features to category to better optimize for memory usage and computation, however we can only do this after the nan values and other values are imputed.
Unique categorical groupings in the categorical features

Just to get a little perspective on how many categories there are for the dataset, we will print each of their unique values for the first several features. This will help us have an idea of how much the dataset will ballon once we onehot encode for each categorical feature value.

# unique values for each categorical feature. need to know how what encoding scheme we should use

​

num_cat_gtoe_10 = []

for column in categorical_df.columns:

    if len(categorical_df[column].unique()) >= 10:

        print(f"######### {column} ##########")

        print(categorical_df[column].unique()) # returns np.ndarray

        print("Number of unique values: {}\n".format(len(categorical_df[column].unique())))

        num_cat_gtoe_10.append(column)

print("There are {} categorical columns with more than 10 categories".format(len(num_cat_gtoe_10)))

Impressions on missing categorical data

Proportionally, there are more missing numerical feature values than there are categorical feature values. There are some categorical feature values that use numeric values to indicate a discrete category or range. This can be gleened by particular values such as '9999' or '999' to indicate a missing value or distinct category. Some of the features that encode their values as such include:

    reg_tl2
    lng_home
    ageg10lfs
    ageg10lfs_t

These features are seemingly identical, and therefore we select one to keep and we drop the other:

    ageg10lfs
    ageg10lfs_t

One feature has 946 unique values which probably indicates that the values maybe numeric in nature as opposed to categorical. There are also two alphabetical letter values feature, namely : 'C' and 'G'. These values will be imputed as np.nan values before the the numeric casting of the feature column. Additionally, the numeric values in this feature are actually strings and will have to be forced cast to int or float values

    v71

Another categorical feature uses extensive numerical coding, interleaved with few alphabetic numeric coding. The instict is to cast the alphabetical coding can be cast to np.nan before the entire feature category can be cast as a numeric column, however, after looking at the value_counts for each of categories in this feature, we can decide to leave it alone as there are a signficant numer of data points that use the alphabetical encoding:

    isic2c

Can be cast to numeric:

    isic2l

This feature seems to be a more granular subfeature of 'cntryrgn' and a few other of the country based features, however this feature has significantly more unique values (176 unique values). As it is a sparser representation of the other superfeatures, it can be dropped:

    reg_tl2

There are a lot of missing values for the lng_home feature when we look at the value_counts. If questionnaire was completed in English, and the language of the exercise was done in English, then may be safe to assume that the primary language spoken at home maybe the same language. Therefore, we will apply this logic for all the missing values in the lng_home feature. These are the language categorical variables.

    lng_bq
    lng_ci
    lng_home

Some features have exactly only one categorical value. These can be dropped as they do no add further information

    uni -> only value is 'cl3770'

Instead of 'No' as the other binary value in a categorical feature, this feature has NaN instead... This value will have to be replaced with the string 'No':

    v270

Some features have a large category space, which would balloon the dataset once onehot-encoded. There are 24 columns with more than 10 categories and they are:

    ['isic1l', 'ageg5lfs', 'v71', 'earnmthalldcl', 'earnhrdcl', 'ctryqual', 'cntryid', 'reg_tl2', 'cnt_brth', 'isic1c', 'lng_ci', 'cntryid_e', 'v92', 'v59', 'edcat8', 'v31', 'lng_bq', 'v19', 'isic2l', 'lng_home', 'birthrgn', 'v212', 'earnhrbonusdcl', 'isic2c']

There are also many seemingly highly correlated and repeated features that can be dropped which pertain to information about which country or region an employee is from:

    'ctryrgn', 'ctryqual', 'birthrgn', 'cntryid_e', 'cntryid'

Our best bet would be to select to keep just one of these features - preferrably a feature with a relatively low number of categories, and then drop the rest.

Therefore we will try to do our best to reduce the number of categories by possibly coming up with higher level categories to encompass these features or cast them as numeric features if the majority of their values are numerically encoded.
Pre-Imputer Imputation for categorical features

Therefore, some of the imputation and preprocessing can be done before the encoding step of the preprocessing portion of the pipeline (dropping,casting, merging, etc). The next few segments of code will go through the next preprocessing steps as outlined above.

# NaN's -> 'No' for 'v270'

​

values = {'v270': 'No'}

categorical_df.fillna(value=values, inplace=True)

print(categorical_df['v270'].unique())

# address missing values for lng_home  and the other language features

​

categorical_df['lng_home'] = categorical_df['lng_bq'].where(categorical_df['lng_bq'] == categorical_df['lng_ci'])

​

# language features

import re

language = ['lng_bq', 'lng_ci', 'lng_home']

num_pattern = re.compile('\d\d\d')

​

# replace the 999 value with np.nan

for col in language:

    categorical_df[col].replace(to_replace=num_pattern, value=np.nan, inplace=True)

# categorical features with only one value

​

for column in categorical_df.columns:

    if len(categorical_df[column].unique()) <=1:

        print(column)

        print("The only unique value for {} is: {}".format(column, categorical_df[column].unique()))

​

# dropping column 'uni' with value only cl3770

categorical_df.drop(columns='uni', inplace=True, axis=1)

# addressing the missing values of feature reg_tl2

​

categorical_df.drop(columns='reg_tl2', inplace=True, axis=1)

# from here, we select only one of these to keep and we will drop the rest

​

country_features = ['ctryrgn', 'ctryqual', 'birthrgn', 'cntryid_e', 'cntryid']

# categorical_df[country_features].head()

​

# we decide to only keep the ctryrgn and drop the rest for now, however, cntryid is another candidate to keep if we wanted more granularity with respect to country information

categorical_df.drop(columns=['ctryqual', 'birthrgn', 'cntryid_e', 'cntryid'], inplace=True, axis=1)

# we drop 'ageg10lfs_t' because it contains identical values to 'ageg10lfs'

​

categorical_df.drop(columns='ageg10lfs_t', inplace=True, axis=1)

# drop dupe education columns

​

categorical_df.drop(columns=['v59', 'v212'], inplace=True, axis=1)

# drop neet because only one value

​

categorical_df.drop('neet', inplace=True, axis=1)

Setting Ordinality to Categorical Features

Although we touched on setting the ordinality of some of the object or categorical features earlier, now would be the time to go through the rest of the categorical features and set the ordinality for them. Unfortunately, unless you do some NLU, and NLP with another feature, we may have to go through each of the features and handcraft ordinality from the text data ourselves.

def categorical_look_filter(dataset):

    """ For each feature in a pandas dataset, print out the unique values in the dataset."""

    

    bin_feats = []

    multicat_feats = []

    

    for col in dataset.columns:

    # Binary

        if 'Yes' in dataset[col].unique() or len(dataset[col].unique()) == 2:

             

            bin_feats.append(col)

    # Multi-Category

        else:

            multicat_feats.append(col)

​

    return bin_feats, multicat_feats

    

bin_features, multicategory_features = categorical_look_filter(categorical_df)

other_bin_features = ["faet12", 'v46', 'earnflag', 'v53', 'nfe12', 'nativelang', 'v205', 'nopaidworkever', 'v84', 'paidwork5', 'fe12', 'paidwork12', 'aetpop']

binary_df = categorical_df[bin_features + other_bin_features]

Encoding Binary Features

We use a combination of pd.Categorical and mappings to encode these features.

# Mapping Binary Cases for these series

​

binary_df = binary_df.replace(to_replace={'Yes': 1, 'No': 0})

binary_df = binary_df.replace(to_replace={'Male': 1, 'Female': 0})

binary_df['faet12'] = binary_df['faet12'].map({'Did not participate in formal AET': 0, 'Participated in formal AET': 1})

binary_df['v46'] = binary_df['v46'].map({'One job or business': 0, 'More than one job or business': 1})

binary_df['earnflag'] = binary_df['earnflag'].map({'Earnings and/or bonuses imputed': 0, 'Reported directly': 1})

binary_df['v53'] = binary_df['v53'].map({'Employee': 0, 'Self-employed': 1})

binary_df['nfe12'] = binary_df['nfe12'].map({'Did not participate in NFE': 0, 'Participated in NFE': 1})

binary_df['nativelang'] = binary_df['nativelang'].map({'Test language not same as native language': 0, 'Test language same as native language': 1})

binary_df['v205'] = binary_df['v205'].map({'Unemployed': 0, 'Employed': 1, 'Out of the labour force': 2})

# These columns are apparently dataframes, so I have to switch over to using the replace method with dictionaries instead

​

binary_df['nopaidworkever'] = binary_df['nopaidworkever'].replace({"Has not has paid work ever": 0, "Has had paid work": 1})

binary_df['v84'] = binary_df['v84'].replace({"Recent work experience in last 12 months": 0, "Currently working (paid or unpaid)": 1})

binary_df['paidwork5'] = binary_df['paidwork5'].replace({"Has not had paid work in past 5 years": 0, "Has had paid work in past 5 years": 1})

binary_df['fe12'] = binary_df['fe12'].replace({"Did not participate in FE": 0, "Participated in FE": 1})

binary_df['paidwork12'] = binary_df['paidwork12'].replace({"Has not had paid work during the 12 months preceding the survey": 0, "Has had paid work during the 12 months preceding the survey": 1})

binary_df['aetpop'] = binary_df['aetpop'].replace({"Excluded from AET population": 0, "AET population": 1})

binary_df.shape

binary_df.head()

Label Encoding

Before you can even impute for the missing values and setting up ordinality with the categorical features, you must first set up the label encoding scheme for each feature. Later, we can use the inverse encoding scheme to set up the ordinality mapping of each categorical feature.

You have to do encoding before you can impute for missing value. Therefore the plan is:

    Use the label encoder to encode the string values in the features (LabelEncoder.fit_transform(X)).
    Impute for the missing values (missing_imputer.fit_transform(X))
    Using the inverse mapping (with LabelEncoder.inverse_transform(X)) for the label encoding, set up the ordinality of the ordinal features using the specified ordinality mapping.

Nominal Cateogrical Encodings

Note: you can get the original encoding scheme back from the encoder with the le.classes_ method, which will return a list of all the unique classes in each feature. These features classes are assigned numeric encodings, based on their index in the alphabetically ordered list of the collection of all the possible classes in that feature.

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

​

nominal_multicategorical_feats = ["v3", 'ctryrgn', 'v91', 'lng_home', 'cnt_brth', 'lng_ci', 'v31', 'v96', "isic1c", "v92", "v88", "v140", "v137"]

nominal_df = categorical_df[nominal_multicategorical_feats]

nominal_categorical_encoding_manifest = {}

​

##### Using LabelEncoding to Encode

# temporarily fill nan values with an encoding, and then after labelencoding, inverse transform, set nans back to np.nan and then impute for missing

nominal_df.fillna('Null', inplace=True)

​

for col in nominal_df.columns:

    le = LabelEncoder()

    le.fit(nominal_df[col].values.ravel())

    nominal_df[col] = le.transform(nominal_df[col].values.ravel())

    null_index = list(le.classes_).index('Null')

    nominal_categorical_encoding_manifest[col] = list(le.classes_)

#     print(f"{col}: {null_index}, type: {type(null_index)}\n")

    nominal_df[col].replace(to_replace=list(le.classes_).index('Null'), value=np.nan, inplace=True)

    le = None


df['edcat7'] = pd.Categorical(df['edcat7'],

                          categories=['Primary or less (ISCED 1 or less)',

                                     'Lower secondary (ISCED 2, ISCED 3C short)',

                                     'Upper secondary (ISCED 3A-B, C long)',

                                     'Post-secondary, non-tertiary (ISCED 4A-B-C)',

                                     'Tertiary – bachelor degree (ISCED 5A)',

                                     r'Tertiary – master/research degree (ISCED 5A/6)\xa0',

                                     'Tertiary - bachelor/master/research degree (ISCED 5A/6)',

                                     'Tertiary – professional degree (ISCED 5B)'],

                          ordered=True)

ordinality_mapping = [

    [["v233", "v280", "v103", "v15", "v24", "v108", "v218", "v171", "v189", \
     "v204", "v166", "v267", "v292", "v155", "v165", "v190", "v288", \
     "v276","v43", "v197", "v214", "v7", "v175", "v139", "v123", "v14", "v178",\
    "v34", "v106", "v246", "v131", "v111", "v173", "v260", "v164", "v186", "v240", "v208",\
    "v275", "v132", "v141", "v25", "v177", "v149", "v23", "v193", "v237", "v162", "v146",\
    "v277", "v40", "v73", "v195"], 
     ['Never','Less than once a month','Less than once a week but at least once a month','At least once a week but not every day','Every day']],
    [['v244', "v65", "v263", "v158", "v57", "v170", "v198", "v278", "v25", "v191", "v114", "v27"], ['Not at all','Very little', 'To some extent', 'To a high extent','To a very high extent']],
    ["ageg10lfs", ['24 or less','25-34', '35-44','45-54', '55 plus']],
    ["v151", ['Aged 15 or younger', 'Aged 16-19', 'Aged 20-24', 'Aged 25-29','Aged 30-34', 'Aged 35 or older']],
    ["v181", ['Extremely dissatisfied', 'Dissatisfied', 'Neither satisfied nor dissatisfied', 'Satisfied', 'Extremely satisfied']],
    ["v271", ['Straightforward','Moderate','Complex']],
    ["v122", ['No', 'Yes, unpaid work for family business', 'Yes, paid work one job or business','Yes, paid work more than one job or business or number of jobs/businesses missing']], 
    [["v247", "v134", "v13", "v18", "v26", "v124", "v99", "v282", "v51", "v2", "v248"], ['Never','Rarely','Less than once a week' ,'At least once a week']], 
    [["v291", "v77"], ['None of the time', 'Up to a quarter of the time','Up to half of the time','More than half of the time','All of the time']],
    ["v269", ['Not useful at all', 'Somewhat useful' , 'Moderately useful','Very useful']],
    [["v216", "v124"], ['Rarely or never','Less than once a week', 'At least once a week']],
    [["v253", "v284"], ['Never', 'Rarely', 'Less than once a week but at least once a month', 'At least once a week']],
    ["ageg5lfs", ['Aged 16-19','Aged 20-24','Aged 25-29','Aged 30-34','Aged 35-39', 'Aged 40-44', 'Aged 45-49','Aged 50-54','Aged 55-59','Aged 60-65']],
    ["v289", ['No income', 'Lowest quintile','Next lowest quintile','Mid-level quintile', 'Next to highest quintile' ,'Highest quintile']],
    ["v261", ['0 - 20 hours','21 - 40 hours', '41 - 60 hours' , '61 - 80 hours', '81 - 100 hours', 'More than 100 hours']],
    ["monthlyincpr", "yearlyincpr"], ['Less than 10','10 to less than 25', '25 to less than 50', '50 to less than 75', '75 to less than 90', '90 or more']],
    ["v221", ['None','Less than 1 month','1 to 6 months','7 to 11 months', '1 or 2 years','3 years or more']],
    [["v85", "v50", "v69"], ['Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree']],
    [["readhome_wle_ca", "influence_wle_ca", "readytolearn_wle_ca", "learnatwork_wle_ca", "readwork_wle_ca", "planning_wle_ca", "taskdisc_wle_ca"], ['All zero response', 'Lowest to 20%', 'More than 20% to 40%', 'More than 40% to 60%', 'More than 60% to 80%', 'More than 80%']],
    [["v82", "v70"], ['Employee, not supervisor', 'Self-employed, not supervisor','Employee, supervising fewer than 5 people', 'Employee, supervising more than 5 people', 'Self-employed, supervisor']],
    ["v200", ['Not definable', 'Less than high school', 'High school' 'Above high school']],
    ["v62", ['A higher level would be needed', 'This level is necessary', 'A lower level would be sufficient']],
    ["v236", ['No, not at all', 'There were no such costs', 'No employer or prospective employer at that time' ,'Yes, partly', 'Yes, totally']],
    ["v19", ['Aged 19 or younger', 'Aged 20-24', 'Aged 25-29', 'Aged 30-34' ,'Aged 35-39' ,'Aged 40-44', 'Aged 45-49', 'Aged 50-54', 'Aged 55 or older']],
    [["earnhrbonusdcl", "earnhrdcl", "earnmthalldcl"], ['Lowest decile','2nd decile', '3rd decile', '4th decile', '5th decile', '6th decile', '7th decile', '8th decile', '9th decile', 'Highest decile']],
    ["imyrcat", ['In host country 5 or fewer years', 'In host country more than 5 years', 'Non-immigrants']],
    ["v48", ['1 to 10 people', '11 to 50 people', '51 to 250 people', 'More than 1000 people', '251 to 1000 people']],
    ["v47", ['Days', 'Weeks',  'Hours']],
    ["iscoskil4", ['Elementary occupations', 'Skilled occupations','Semi-skilled blue-collar occupations', 'Semi-skilled white-collar occupations']],
    ["v94", ['Respondent reported no learning activities', 'Respondent reported 1 learning activity', 'Respondent reported learning activities but number is not known', 'Respondent reported more than 1 learning activity']],
    ["v8", ['Decreased', 'Stayed more or less the same', 'Increased']],
    ['edcat7', ['Primary or less (ISCED 1 or less)', 
                'Lower secondary (ISCED 2, ISCED 3C short)', 
                'Upper secondary (ISCED 3A-B, C long)', 
                'Post-secondary, non-tertiary (ISCED 4A-B-C)', 
                'Tertiary – bachelor degree (ISCED 5A)', 
                r'Tertiary – master/research degree (ISCED 5A/6)\xa0', 
                'Tertiary - bachelor/master/research degree (ISCED 5A/6)', 
                'Tertiary – professional degree (ISCED 5B)']
    ]

]


ordinal_features = []
for mask, cat in ordinality_mapping:
    if isinstance(mask, list):
        for label in mask:
#             print(f"{label}: {cat}")
            ordinal_features.append(label)

​

    elif isinstance(mask, str):

#         print(f"{mask}: {cat}")

        ordinal_features.append(mask)

​

ordinal_df = categorical_df[ordinal_features]

ordinal_df.fillna('Null', inplace=True)

ordinal_df.shape

# mapping the values based on the ordinality mapping

# some of these features, for some reason, are dataframes, so you have to convert them to series as an intermediary

# and then do inplace replacement of the string values with the mapping.

# the integere mapping here implies ordinality, where as the integer label encoding for the nominal features doesn't

​

for mask, cat in ordinality_mapping:

    if isinstance(mask, list):

        for label in mask:

            indiv_feat_mapping = {key: val for val, key in enumerate(cat)}

            series = pd.Series(ordinal_df[label].values.ravel())

            ordinal_df[label] = series.map(indiv_feat_mapping)

​

    elif isinstance(mask, str):

        indiv_feat_mapping = {key: val for val, key in enumerate(cat)}

        series = pd.Series(ordinal_df[mask].values.ravel())

        ordinal_df[mask] = series.map(indiv_feat_mapping)

# look at num missing values

​

ordinal_features_to_drop = ["v278", "v25", "v124", "v25", "v124", "v200"]

(ordinal_df[ordinal_features_to_drop].isnull().sum()/ordinal_df[ordinal_features_to_drop].shape[0] * 100).sort_values(ascending=False)

# The above features have too many features missing, so we will drop them

​

ordinal_df.drop(ordinal_features_to_drop, inplace=True, axis=1)

ordinal_df.shape

ordinal_df.head()

Putting all the hard work of preprocessing our categorical features together back into one dataframe

categorical_df = pd.concat([binary_df, ordinal_df, nominal_df], axis=1)

categorical_df.shape

categorical_df.head()

Imputation for Missing Values

We want to then use an informed approach to imputation of missing data values. One method could be the use of KNN imputation or random forest and proximity matrix to imputing the missing values for our dataset with some similarity metric.

We need to import for missing values first because we can apply the astype 'category' to these categorical, and oridinal features.

# !pip install missingpy

We combine the numeric dataframe back with the categorical dataframe at this point.

final_df = pd.concat([numeric_df, categorical_df], axis=1)

Final bit of preprocessing

occupational_feats = ['v224', 'isco2c']

values_to_replace = [9995, 9996, 9997, 9998, 9999]

​

for col in occupational_feats:

    for val in values_to_replace:

        final_df.loc[final_df[col] == val, col] = np.nan

final_df.drop('v224', inplace=True, axis=1)

((final_df.isnull()/final_df.shape[0]) * 100).sum().sort_values(ascending=False)[:5]

​

# Split into features and class

​

X = final_df.loc[:, final_df.columns != 'job_performance'].values

y = final_df['job_performance'].values

!pip install missingpy

You could use either Missing Forest Imputation or KNN imputation. We decide to go with KNN imputation instead.

# Let X be an array containing missing values

# from missingpy import MissForest

# imputer = MissForest()

# X_imputed = imputer.fit_transform(X)

# missing value imputation with KNN

​

from timeit import default_timer

from missingpy import KNNImputer, MissForest

​

knn = KNNImputer(n_neighbors=3, weights="uniform",

                 metric="masked_euclidean", row_max_missing=0.8,

                 col_max_missing=0.8, copy=True)

​

start_timer = default_timer()

knn_missing_imputation = knn.fit_transform(X)

end_timer = default_timer()

print("{}s".format((end_timer - start_timer)/10))

# we see that the imputation actually works

​

assert len(knn_missing_imputation) == final_df.shape[0]

#The imputation goes back into the dataframe

​

final_df_imputed = pd.DataFrame(knn_missing_imputation, columns = final_df.columns[final_df.columns != 'job_performance'])

final_df_imputed['job_performance'] = pd.Series(y)

final_df_imputed.head()

# new set of ordinal features after some of them have beend dropped

# we won't have to do anything with ordinal features again

​

ordinal_features = list(set(ordinal_features) - set(['v124', 'v200', 'v25', 'v278']))

One Hot Encoding for Nominal Features

We had to temporarily label encoded the nominal categorical features becuase

final_df_imputed[nominal_multicategorical_feats]= final_df_imputed[nominal_multicategorical_feats].applymap(lambda x: np.round(x)).astype('int64')

Reverse the labeling before the onehotencoding so that we can tell what the groupings are from the column headers after 1HE.

from sklearn.preprocessing import LabelEncoder

​

final_df_imputed[nominal_multicategorical_feats].head()

# we use the dictionary of kvps for the nominal categorical values we created earlier

np.random.seed(123)

​

null_dict = {}

​

assert len(nominal_multicategorical_feats) == len(nominal_categorical_encoding_manifest)

​

for key, value in nominal_categorical_encoding_manifest.items():

    null_dict[key] = value.index('Null')

    null_replacement = value.index('Null')

    

    while null_replacement == value.index('Null'):

        null_replacement = np.random.randint(low=0, high=len(value))                                

        

    final_df_imputed.loc[final_df_imputed[key] == value.index('Null'), key] = null_replacement                       

# ensure that the replacement for the null value of the value pair holds true to this condition

​

for col, null_value in null_dict.items():

    assert null_value not in final_df_imputed[col].unique(), "The previous null value encoding has not yet been replaced with a new random value encoding"

# now retrieve the string value from the values list of the nominal_categorical_encoding_manifest dictionary

# and replace it with the corresponding indexed encoding, and then apply the onehotencoding on these values

​

start_time = default_timer()

​

for key, values in nominal_categorical_encoding_manifest.items():

    for value in values:

        final_df_imputed.loc[final_df_imputed[key] == values.index(value), key] = values[values.index(value)]

        

end_time = default_timer()

​

print("The replacement operation took {}s".format((end_time-start_time)/1000))

final_df_imputed[nominal_categorical_encoding_manifest.keys()].head()

# Using Dummies to OneHotEncoder, usable code snippet for dummies

# dummy_na=True provides an extra column for the nan values, which are 1HE

for col in nominal_categorical_encoding_manifest.keys():

    final_df_imputed = pd.concat([final_df_imputed, pd.get_dummies(final_df_imputed[col], prefix=col, dummy_na=False)],axis=1).drop([col],axis=1)

final_df_imputed.shape

final_df_imputed.head(3)

# Save out the dataset to a csv

import os

​

out_csv_name = './post-processed-employee-performance-dataset.csv'

​

if not os.path.exists(out_csv_name):

    final_df_imputed.to_csv('./post-processed-employee-performance-dataset.csv')

Feature Selection: Forward Selection

We enter our feature selection step of the pipeline and we use an out of the box forward feature selection method from ML XTend to pick the features that result in the best predictive ability of our models. There are a number of different feature selection techniques that we could use (and we may explore that in the future), but for now, using a simple forward selection process is what we are going to stick with.

!pip install mlxtend

Test Bed

from sklearn.model_selection import train_test_split

​

# remmeber to sample down your dataset for testing

np.random.seed(123)

test = final_df_imputed.copy().sample(frac=0.10)

if not os.path.exists('./post-processed-employee-performance-dataset_sample.csv'):

    test.to_csv('./post-processed-employee-performance-dataset_sample.csv')

​

# Train/test split

X_train, X_test, y_train, y_test = train_test_split(

    test.loc[:, test.columns != 'job_performance'].values,

    test['job_performance'].values.ravel(),

    test_size=0.25,

    random_state=42)

​

y_train = y_train.ravel()

y_test = y_test.ravel()

catch up with Charlie, apply, dip into sharpest minds, cold email template, leet code. following up and sending out cold emails. now is the best time to apply.

from mlxtend.feature_selection import SequentialFeatureSelector as sfs

from mlxtend.feature_selection import ColumnSelector, ExhaustiveFeatureSelector

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.pipeline import make_pipeline

​

# # Build RF regressor to use in feature selection

rfr = RandomForestRegressor(n_estimators=10, n_jobs=-1)

​

# # Build step forward feature selection

efs1 = ExhaustiveFeatureSelector(rfr, 

                                 min_features=1, 

                                 max_features=round(test.shape[1]/10), 

                                 print_progress=True, 

                                 scoring='r2', # mean_squared_error

                                 cv=10, 

                                 n_jobs=1, 

                                 pre_dispatch=None, # '2n_jobs' 

                                 clone_estimator=True)

​

# efs1.attrs: {best_idx_, best_feature_names_ , best_score_ , subsets_ }

​

# # Build step forward feature selection

# sfs1 = sfs(rfr,

#            k_features=round(test.shape[1]/10),

#            forward=True,

#            floating=False,

#            verbose=2,

#            pre_dispatch=None, 

#            n_jobs=-1,

#            scoring=mean_squared_error,  # r2

#            cv=10)

​

# sfs1.attrs: {k_feature_idx_, k_feature_names_ , _ , subsets_, custom_feature_names }

​

#  Perform Feature Selection

start_time = default_timer()

hist = efs1.fit(X_train, y_train)

efs1_pred = efs1.predict(X_test, y_test)

end_time = default_timer()

print("Job time: {}s".format(end-time - start_time/1000))

break

Plot the results of the forward feature selection

rom mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

import matplotlib.pyplot as plt

​

sfs = SFS(knn, 

          k_features=4, 

          forward=True, 

          floating=False, 

          scoring='accuracy',

          verbose=2,

          cv=5)

​

sfs = sfs.fit(X, y)

​

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

​

plt.ylim([0.8, 1])

plt.title('Sequential Forward Selection (w. StdDev)')

plt.grid()

plt.show()

!pip install tpot

TPOT Model Performance

Teapot does hyperparameter tuning, model selection and preprocessing all in one pipeline. It takes a while to train the say the least. We will select the best parameter from training on the null dataset (without any feature engineering) to see if we can get a functioning model from this exercise

import timeit

from tpot import TPOTClassifier

from sklearn.metrics import (

    confusion_matrix,

    roc_auc_score,

    precision_recall_fscore_support,

    accuracy_score,

)

from pprint import pprint

​

​

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, n_jobs=-1)

​

prec_rec_fsc_sup = ["precision", "recall", "fscore", "support"]

​

start_time = timeit.default_timer()

tpot.fit(X_train, y_train)

y_pred = tpot.predict(X_test)

end_time = timeit.default_timer()

runtime = end_time - start_time

print(f"Total runtime for the {name} dataset: {runtime}s")

​

​

print("\nConfusion Matrix for the {} dataset\n{}\n".format(confusion_matrix(name, y_test, y_pred)))

​

print("Precision/Recall/FScore/Support for the {} dataset".format(name))

for met, val in zip(prec_rec_fsc_sup, precision_recall_fscore_support(y_test, y_pred)):

    pprint("{}: {}".format(met, val))

​

print("Accuracy score for the {} dataset: {}".format(name, accuracy_score(y_test, y_pred)))

Hyperparameter Tuning

After a while, you start getting an intuition of which hyperparameters are more important than others.

# grid search, beam search, bayesian optimization, random search

Updating preview...
