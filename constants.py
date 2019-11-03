# if using google colab use DRIVENAME
DRIVENAME = '/content/drive/'
# FILENAME = "/content/drive/My Drive/sharpestminds_dataset/hw5-trainingset-cl3770.csv"
FILENAME = "/home/dennis/PythonProjects/datasets/sharpest-minds/hw5-trainingset-cl3770.csv"

CONSIDERED_MISSING_VALUES = ['999', 9995, '9995', 9996, '9996', 9997, "9997", 9998, '9998', 9999, '9999', '99999']
REDUNDANT_FEATURES = ["readytolearn_wle_ca", "icthome_wle_ca", "ictwork_wle_ca", 
                      "influence_wle_ca", "planning_wle_ca", "readhome_wle_ca", 
                      "readwork_wle_ca", "taskdisc_wle_ca", "writhome_wle_ca", 
                      "writwork_wle_ca", "ageg10lfs", "ageg10lfs_t", "edcat7", 
                      "edcat8", "isco2c", "isic2c", "isic2l","earnflag", "reg_tl2", "lng_bq", 
                      "lng_ci", "edlevel3", "nfehrsnjr", "nfehrsjr", "nfe12jr", 
                      "fnfe12jr", "fnfaet12jr", "faet12jr", "faet12njr", "fe12", "monthlyincpr",
                      "earnhrdcl", "earnhrbonusdcl", "row", "uni", "cntryid_e", "v270", 
                      "v205", "neet", "v84", "nfe12njr", "fnfaet12njr"
                     ]

ORDINAL_VARIABLE_MAPPING = [
    [["v233", "v280", "v103", "v15", "v24", "v108", "v218", "v171", "v189", 
     "v204", "v166", "v267", "v292", "v155", "v165", "v190", "v288", 
     "v276","v43", "v197", "v214", "v7", "v175", "v139", "v123", "v14", "v178",
    "v34", "v106", "v246", "v131", "v111", "v173", "v260", "v164", "v186", "v240", "v208",
    "v275", "v132", "v141", "v25", "v177", "v149", "v23", "v193", "v237", "v162", "v146",
    "v277", "v40", "v73", "v195"], 
    ['Never', 'Less than once a month','Less than once a week but at least once a month','At least once a week but not every day','Every day']],
    [['v244', "v65", "v263", "v158", "v57", "v170", "v198", "v191", "v114", "v27"], ['Not at all', 'Very little', 'To some extent', 'To a high extent','To a very high extent']], 
    [["v151"], ['Aged 15 or younger', 'Aged 16-19', 'Aged 20-24', 'Aged 25-29','Aged 30-34', 'Aged 35 or older']],
    [["v181"], ['Extremely dissatisfied', 'Dissatisfied', 'Neither satisfied nor dissatisfied', 'Satisfied', 'Extremely satisfied']],
    [["v271"], ['Straightforward','Moderate','Complex']], 
    [["v247", "v134", "v13", "v18", "v26", "v124", "v99", "v282", "v51", "v2", "v229", "v248"], ['Never','Rarely or never', 'Rarely','Less than once a week', 
                                                                                         'Less than once a week but at least once a month' ,'At least once a week']],
    [["v85", "v50", "v69"], ['Strongly disagree', 'Disagree', 'Neither agree nor disagree', 'Agree', 'Strongly agree']],
    [["v291", "v77"], ['None of the time', 'Up to a quarter of the time','Up to half of the time','More than half of the time','All of the time']],
    [["v269"], ['Not useful at all', 'Somewhat useful' , 'Moderately useful','Very useful']],
    [["v216"], ['Rarely or never','Less than once a week', 'At least once a week']],
    [["v253", "v278", "v284"], ['Never', 'Rarely', 'Less than once a month', 'Less than once a week but at least once a month', 
                        'At least once a week', 'At least once a week but not every day', 'Every day']],
    [["ageg5lfs"], ['Aged 16-19','Aged 20-24','Aged 25-29','Aged 30-34','Aged 35-39', 'Aged 40-44', 'Aged 45-49','Aged 50-54','Aged 55-59','Aged 60-65']],
    [["v289"], ['No income', 'Lowest quintile','Next lowest quintile','Mid-level quintile', 'Next to highest quintile' ,'Highest quintile']],
    [["v261"], ['0 - 20 hours','21 - 40 hours', '41 - 60 hours' , '61 - 80 hours', '81 - 100 hours', 'More than 100 hours']],
    [["v221"], ['None','Less than 1 month','1 to 6 months','7 to 11 months', '1 or 2 years','3 years or more']],
    [["v82", "v70"], ['Self-employed or unpaid family worker', 'Employee, not supervisor', 'Self-employed, not supervisor',
                      'Employee, supervising fewer than 5 people', 'Employee, supervising more than 5 people', 'Self-employed, supervisor']],
    [["v200"], ['Not definable', 'Less than high school', 'High school', 'Above high school']],
    [["v62"], ['A higher level would be needed', 'This level is necessary', 'A lower level would be sufficient']],
    [["v236"], ['No, not at all', 'There were no such costs', 'No employer or prospective employer at that time' ,'Yes, partly', 'Yes, totally']],
    [["v19"], ['Aged 19 or younger', 'Aged 20-24', 'Aged 25-29', 'Aged 30-34' ,'Aged 35-39' ,'Aged 40-44', 'Aged 45-49', 'Aged 50-54', 'Aged 55 or older']],
    [["imyrcat"], ['In host country 5 or fewer years', 'In host country more than 5 years', 'Non-immigrants']],
    [["v48"], ['1 to 10 people', '11 to 50 people', '51 to 250 people', 'More than 1000 people', '251 to 1000 people']],
    [["v47"], ['Days', 'Weeks',  'Hours']],
    [["iscoskil4"], ['Elementary occupations', 'Skilled occupations','Semi-skilled blue-collar occupations', 'Semi-skilled white-collar occupations']],
    [["v94"], ['Respondent reported no learning activities', 'Respondent reported 1 learning activity', 
               'Respondent reported learning activities but number is not known', 'Respondent reported more than 1 learning activity']],
    [["v8"], ['Decreased', 'Stayed more or less the same', 'Increased']],
    [['edcat6'], ['Lower secondary or less (ISCED 1,2, 3C short or less)\xa0',
                'Upper secondary (ISCED 3A-B, C long)',
                'Post-secondary, non-tertiary (ISCED 4A-B-C)',
                'Tertiary – bachelor degree (ISCED 5A)',
                'Tertiary - bachelor/master/research degree (ISCED 5A/6)',
                'Tertiary – master/research degree (ISCED 5A/6)',
                'Tertiary – professional degree (ISCED 5B)']]
]

BINARY_VARIABLE_MAPPING = {
    "gender_r": {'Male': 0, 'Female': 1},
    "faet12": {'Did not participate in formal AET': 0, 'Participated in formal AET': 1},
    "v46" : {'One job or business': 0, 'More than one job or business': 1},
    "v53" : {'Employee': 0, 'Self-employed': 1},
    "nfe12" : {'Did not participate in NFE': 0, 'Participated in NFE': 1},
    "nativelang" : {'Did not participate in NFE': 0, 'Participated in NFE': 1},
    "nopaidworkever": {"Has not has paid work ever": 0, "Has had paid work": 1},
    "paidwork5" : {"Has not had paid work in past 5 years": 0, "Has had paid work in past 5 years": 1},
    "paidwork12" : {"Has not had paid work during the 12 months preceding the survey": 0, "Has had paid work during the 12 months preceding the survey": 1},
    "aetpop" : {"Excluded from AET population": 0, "AET population": 1},
    # "edwork" : {"In work only": 0, "In education and work": 1},
    # "v122" : {'Yes, unpaid work for family business': 0, 'Yes, paid work one job or business': 1, 'Yes, paid work more than one job or business or number of jobs/businesses missing': 2},
    "nativelang": {'Test language not same as native language': 0, 'Test language same as native language':1},
    "fnfaet12": {'Did not participate in formal or non-formal AET': 0, 'Participated in formal and/or non-formal AET': 1}
}

ORDINAL_FEATURE_NAMES = [
    "v233", "v280", "v103", "v15", "v24", "v108", "v218", "v171", "v189",
     "v204", "v166", "v267", "v292", "v155", "v165", "v190", "v288",
     "v276","v43", "v197", "v214", "v7", "v175", "v139", "v123", "v14", 
     "v178", "v34", "v106", "v246", "v131", "v111", "v173", "v260", "v164", 
     "v186", "v240", "v208", "v275", "v132", "v141", "v25", "v177", "v149", 
     "v23", "v193", "v237", "v162", "v146", "v277", "v40", "v73", "v195", 'v244',
     "v65", "v263", "v158", "v57", "v170", "v198", "v278", "v191", "v114", "v27", 
     "v151", "v181", "v271", "v247", "v134", "v13", "v18", "v26", "v124", "v99", 
     "v282", "v51", "v2", "v229", "v248","v291", "v77","v269", "v216","v253", 
     "v284", "ageg5lfs", "v289", "v261", "v221", "v85","v50","v69", "v82", 
     "v70", "v200", "v62", "v236","v19", "imyrcat","v48","v47","iscoskil4","v94",
     "v8",'edcat6',]

NOMINAL_FEATURE_NAMES = [
  "cntryid", "lng_home", "cnt_h", "cnt_brth", 
  "ctryqual", "birthrgn", "ctryrgn", "isic1c",
  "isic1l", "v31", "v137", "v234", "v91","v92",
  "v88", "v140", "v3",]

POSTPROCESSED_DATAPATH = "/home/dennis/PythonProjects/datasets/sharpest-minds/post_processed_employee_dataset_10_feats_sfs.csv"
PERIODIC_CHECKPOINT_FOLDER = "/home/dennis/Desktop/Link to datascience_job_portfolio/sharpestminds_project/periodic_checkpoint_folder"

