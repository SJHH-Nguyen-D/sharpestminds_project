import re

RAWDATASETPATH = "/home/dennis/PythonProjects/datasets/hw5-trainingset-cl3770.csv"
POSTPROCESSED_DATAPATH = "/home/dennis/PythonProjects/datasets/sharpest-minds/post_processed_employee_dataset_10_feats_sfs.csv"
PERIODIC_CHECKPOINT_FOLDER = "/home/dennis/Desktop/Link to datascience_job_portfolio/sharpestminds_project/periodic_checkpoint_folder"
FEATURES_TO_DROP = [
    "earnhrbonusppp",
    "earnhrbonus",
    "earnhrppp",
    "earnhr",
    "earnmthbonusppp",
    "earnmthselfppp",
    "earnmthbonus",
    "earnmthppp",
    "earnmth",
    "earnmthallppp",
    "earnmthall",
    "fnfaet12njr",
    "fnfaet12jr",
    "fnfaet12",
    "fnfe12jr",
    "edcat8",
    "edwork",
    "edcat6",
    "edlevel3",
    "icthome_wle_ca",
    "ictwork_wle_ca",
    "nfe12jr",
    "nfe12njr",
    "nfehrsjr",
    "nfehrsnjr",
    "writhome_wle_ca",
    "writwork_wle_ca",
    "yrsget",
    "yrsqual",
    "isco2l",
    "isco1l",
    "isco1c",
    "isic2l",
    "isic1l",
    "isic2c",
    "v224",
    "v278",
    "v25",
    "v124",
    "v25",
    "v124",
    "v200",
    "neet",
    "v59",
    "v212",
    "ctryqual",
    "birthrgn",
    "cntryid_e",
    "cntryid",
    "ageg10lfs_t",
    "age_r",
    "row",
    "v239",
    "uni",
    "reg_tl2",
]


ORDINALITY_MAPPING = [

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

    [["monthlyincpr", "yearlyincpr"], ['Less than 10','10 to less than 25', '25 to less than 50', '50 to less than 75', '75 to less than 90', '90 or more']],

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
