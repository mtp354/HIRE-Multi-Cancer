# Configurations of the multi-cancer model
import numpy as np
import pandas as pd
import glob
import os
import random

# Aesthetic Preferences
np.set_printoptions(precision=5, suppress=True)

MODE = 'visualize'
# Options:
# - calibrate: run simulated annealing for cancer incidence (one site)
# - visualize: plot incidence and mortality, output cancer incidence, cancer count, alive count
# - cancer_dist: plot cancer pdf and cdf
SAVE_RESULTS = True  # whether to save results to file
SOJOURN_TIME = False
SMOOTH_CALIBRATION_PLOTS = True # whether to save plots from each calibration iteration
# Define cohort characteristics
COHORT_YEAR = 1957  # birth year of the cohort
START_AGE = 0
END_AGE = 100
COHORT_SEX = 'Male'  # Female/Male
COHORT_RACE = 'White'  # Black/White
NUM_PATIENTS = 100_000
CANCER_SITES = ['Colorectal']
# Full list:
# MP 'Bladder' 'Breast' 'Cervical' 'Colorectal' 'Esophageal' 
# JP 'Gastric' 'Lung' 'Prostate' 'Uterine'
# FL 'Pancreas' 'Ovarian' 'Kidney' 'Brain' 'Liver' 'Gallbladder'

# Raise exceptions for male/ovarian, male/uterine, male/cervical, female/prostrate
if COHORT_SEX == 'Male' and ('Ovarian' in CANCER_SITES or 'Uterine' in CANCER_SITES or 'Cervical' in CANCER_SITES):
    raise Exception("Cancer site and cohort sex combination is not valid in configs.py")
elif COHORT_SEX == 'Female' and 'Prostate' in CANCER_SITES:
    raise Exception("Cancer site and cohort sex combination is not valid in configs.py")

# Raise exceptions for other modes of the model
if len(CANCER_SITES) > 1 and MODE == 'calibrate':
    raise Exception("You cannot calibrate multiple cancer sites at the same time")
if len(CANCER_SITES) > 1 and MODE == 'cancer_dist':
    raise Exception("You can only run cancer_dist for one cancer site")

# Define multiprocessing parameters
NUM_PROCESSES = 10

# Define simulated annealing parameter
NUM_ITERATIONS = 1_000
START_TEMP = 10
STEP_SIZE = 0.001 #0.001
VERBOSE = True
MASK_SIZE = 0.5 # value between 0 and 1, the fraction of values to modify each step
LOAD_LATEST = True# If true, load the latest cancer_pdf from file as starting point
# LOAD_LATEST is used to get the most recently calibrated numpy file to run the model
# First checks if there is a previous file for same sex/race/cancer site, then same sex/cancer site,
# then same race/cancer site, then same cancer site

# Note: You can either do multi-calibration by increasing or decreasing cohort years
# Range is based on FIRST_COHORT and LAST_COHORT
# To do calibration in ascending cohort years, you MUST have a starting numpy file for the FIRST_COHORT or else
# LOAD_LATEST cannot work with MULTI_COHORT_CALIBRATION correctly
# So generally you should set both LOAD_LATEST and MULTI_COHORT_CALIBRATION to True
# You MUST do multi-calibration in ascending order FIRST before doing descending order
# You CANNOT start multi-cohort calibration in descending order first
# When you do reverse calibration, remember that the LAST_COHORT looks at the next +1 birth year cohort year
MULTI_COHORT_CALIBRATION = False
REVERSE_MULTI_COHORT_CALIBRATION = False # determines whether you want to reverse the cohort year range in calibration
if MULTI_COHORT_CALIBRATION:
    FIRST_COHORT = 1935
    LAST_COHORT = 1965
if REVERSE_MULTI_COHORT_CALIBRATION == True and MULTI_COHORT_CALIBRATION == False:
    raise ValueError("ERROR: You cannot have REVERSE_MULTI_COHORT_CALIBRATION set to True while MULTI_COHORT_CALIBRATION is set to False")
if MULTI_COHORT_CALIBRATION and MODE != "calibrate":
    raise ValueError("ERROR: You cannot set MULTI_COHORT_CALIBRATION to True for a MODE other than calibrate")


# Define input and output paths
PATHS = {
    'incidence': '../data/cancer_incidence/',
    'mortality': '../data/mortality/',
    'survival': '../data/cancer_survival/',
    'calibration': '../outputs/calibration/',
    'plots_calibration': '../outputs/calibration/plots/',
    'sojourn_time': '../data/Sojourn Times/',
    'plots': '../outputs/plots/',
    'output': '../outputs/'
}


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Load in sojorn times
sojourn = pd.read_csv(PATHS['sojourn_time'] + 'Sojourn Estimates.csv')
sj_cancer_sites = {}
for i in range(len(CANCER_SITES)):
    s = sojourn[sojourn['Site'].isin([CANCER_SITES[i]])]
    sj_cancer_sites[i] = np.random.triangular(s['Lower'], s['Sojourn Time'], s['Upper'], NUM_PATIENTS).astype(int)

random_numbers_array = np.random.rand(NUM_PATIENTS, 20)

# Selecting Cohort
def select_cohort(birthyear, sex, race):
    # Load input data
    CANCER_INC = pd.read_csv(f'{PATHS["incidence"]}Incidence.csv')
    CANCER_INC = CANCER_INC[CANCER_INC['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest

    # Load in mortality data
    MORT = pd.read_csv(f'{PATHS["mortality"]}Mortality.csv')
    MORT = MORT[~MORT['Site'].isin(CANCER_SITES)]  # Removing the cancers of interest
    MORT = MORT.groupby(['Cohort','Age','Sex','Race']).agg({'Rate':'sum'}).reset_index()  # Summing over the remaining sites

    # Load in Survival data # TODO: no male breast survival data
    SURV = pd.read_csv(f'{PATHS["survival"]}Survival.csv')  # This is the 10 year survival by cause
    SURV = SURV[SURV['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest

    CANCER_INC.query('Sex == @sex & Race == @race & Cohort == @birthyear', inplace=True)
    
    # Impute SEER data using earlier cohorts upto age 83
    if CANCER_INC['Age'].max() < 83:
        i=1
        while CANCER_INC['Age'].max()<83:
            cohort_year = birthyear-i
            cancer_inc = pd.read_csv(f'{PATHS["incidence"]}Incidence.csv')
            cancer_inc = cancer_inc[cancer_inc['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest
            cancer_inc.query('Sex == @sex & Race == @race & Cohort == @cohort_year', inplace=True)
            CANCER_INC = pd.concat([CANCER_INC, cancer_inc.iloc[-1,:].to_frame().T])
            i = i+1

    # CANCER_INC = CANCER_INC.iloc[:-4,:] # when you need to adjust maximum age
    # Add linear line from anchoring point to the age at first incidence data point
    if list(CANCER_INC['Age'])[0] > 18:
        fillup_age = list(range(18, list(CANCER_INC['Age'])[0]))
        slope = list(CANCER_INC['Rate'])[0]/(list(CANCER_INC['Age'])[0]-18)
        intercept = -18*slope
        fillup_rate = slope*np.array(fillup_age)+intercept
        fillup_df = pd.DataFrame({'Age': fillup_age, 'Rate': fillup_rate})
        CANCER_INC = pd.concat([fillup_df, CANCER_INC])

        min_age = 18
        # max_age = min(2018 - birthyear, 83)
    else:
        # For plotting and objective, we only compare years we have data
        min_age = max(1975 - birthyear, 0)
        # max_age = min(2018 - birthyear, 83)
    # min_age = 18
    max_age = 83 # max_age needs to be 83 as we are imputing SEER data to age 83
    
    MORT.query('Sex == @sex & Race == @race & Cohort == @birthyear', inplace=True)
    SURV.query('Sex == @sex & Race == @race', inplace=True)

    # Create the cdf all-cause mortality
    ac_cdf = np.cumsum(MORT['Rate'].to_numpy())/100000
    # Creating the conditional CDF for all-cause mortality by age (doing this here to save runtime)
    ac_cdf = np.tile(ac_cdf, (END_AGE - START_AGE + 1, 1))  # (current age, future age)
    
    for i in range(END_AGE - START_AGE + 1):
        ac_cdf[i, :] -= ac_cdf[i, i]  # Subtract the death at current age
    ac_cdf[:, -1] = 1.0  # Adding 1.0 to the end to ensure death at 100
    ac_cdf = np.clip(ac_cdf, 0.0, 1.0)
    
    # Load all cancer incidence target data
    if len(CANCER_SITES) == 1: # only 1 cancer site
        CANCER_INC = CANCER_INC['Rate'].to_numpy()
    else: # multiple cancer sites
        # Need to separate each cancer incidence
        # Add numpy array for each cancer site into a lst
        CANCER_INC_lst = []
        for i in range(len(CANCER_SITES)):
            temp = CANCER_INC[CANCER_INC['Site']==CANCER_SITES[i]]
            tempArr = temp['Rate'].to_numpy()
            CANCER_INC_lst.append(tempArr)

    # Loading in cancer pdf, this is the thing that will be optimized over
    CANCER_PDF = 0.002 * np.ones(END_AGE - START_AGE + 1)  # starting from 0 incidence and using bias optimization
    CANCER_PDF[:35] = 0.0

    if LOAD_LATEST:
        if len(CANCER_SITES) == 1: # 1 cancer site
            # Check if there is a previous numpy file matching the same sex and race and cancer site
            list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_SEX}_{COHORT_RACE}_*{CANCER_SITES[0]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same sex and cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_SEX}_*{CANCER_SITES[0]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same race and cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_RACE}_*{CANCER_SITES[0]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{CANCER_SITES[0]}_*.npy')
            if len(list_of_files) == 0:
                raise ValueError("No suitable LOAD_LATEST file, set LOAD_LATEST to FALSE")

            # Look at all the unique cohort years in the file names
            all_cohort_years = []
            for file in list_of_files:
                year = file.split('_')[2] # grabs the cohort year
                if int(year) not in all_cohort_years:
                    all_cohort_years.append(int(year))
            if MULTI_COHORT_CALIBRATION == False or REVERSE_MULTI_COHORT_CALIBRATION == False: # ascending birth year calibration
                # Sort ascending years
                all_cohort_years.sort()
                # Get the max calibrated cohort year that is just below or equal to the COHORT_YEAR
                for year in all_cohort_years:
                    if year <= birthyear:
                        max_year = year
                final_list = []
                for file in list_of_files:
                    if f'_{max_year}_' in file:
                        final_list.append(file)
            elif MULTI_COHORT_CALIBRATION and REVERSE_MULTI_COHORT_CALIBRATION: # descending birth year calibration
                # Get the min calibrated cohort year that is just above the COHORT_YEAR
                min_year = birthyear + 1
                final_list = []
                for file in list_of_files:
                    if f'_{min_year}_' in file:
                        final_list.append(file)
            else:
                raise ValueError("ERROR: LOAD_LATEST fails in configs.py")
            # Read the latest file
            latest_file = max(final_list, key=os.path.getctime)
            CANCER_PDF = np.load(latest_file)
        else: # multiple cancer sites
            CANCER_PDF_lst = []
            for i in range(len(CANCER_SITES)):
                # Check if there is a previous numpy file matching the same sex and race and cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_SEX}_{COHORT_RACE}_*{CANCER_SITES[i]}_*.npy')
                if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same sex and cancer site
                    list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_SEX}_*{CANCER_SITES[i]}_*.npy')
                if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same race and cancer site
                    list_of_files = glob.glob(f'{PATHS["calibration"]}*{COHORT_RACE}_*{CANCER_SITES[i]}_*.npy')
                if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same cancer site
                    list_of_files = glob.glob(f'{PATHS["calibration"]}*{CANCER_SITES[i]}_*.npy')
                if len(list_of_files) == 0:
                    raise ValueError("No suitable LOAD_LATEST file, set LOAD_LATEST to FALSE")

                # Look at all the unique cohort years in the file names
                all_cohort_years = []
                for file in list_of_files:
                    year = file.split('_')[2] # grabs the cohort year
                    if int(year) not in all_cohort_years:
                        all_cohort_years.append(int(year))
                if MULTI_COHORT_CALIBRATION == False or REVERSE_MULTI_COHORT_CALIBRATION == False: # ascending birth year calibration
                    # Sort ascending years
                    all_cohort_years.sort()
                    # Get the max calibrated cohort year that is just below or equal to the COHORT_YEAR
                    for year in all_cohort_years:
                        if year <= birthyear:
                            max_year = year
                    final_list = []
                    for file in list_of_files:
                        if f'_{max_year}_' in file:
                            final_list.append(file)
                elif MULTI_COHORT_CALIBRATION and REVERSE_MULTI_COHORT_CALIBRATION: # descending birth year calibration
                    # Get the min calibrated cohort year that is just above the COHORT_YEAR
                    min_year = birthyear + 1
                    final_list = []
                    for file in list_of_files:
                        if f'_{min_year}_' in file:
                            final_list.append(file)
                else:
                    raise ValueError("ERROR: LOAD_LATEST fails in configs.py")
                # Read the latest file
                latest_file = max(final_list, key=os.path.getctime)
                CANCER_PDF = np.load(latest_file)
                CANCER_PDF_lst.append(CANCER_PDF)

    # Loading in cancer survival data
    if len(CANCER_SITES) == 1: # 1 cancer site
        SURV = SURV[['Cancer_Death','Other_Death']].to_numpy()  # 10 year survival
        SURV = 1 - SURV**(1/10)  # Converting to annual probability of death (assuming constant rate)

        # Converting into probability of death at each follow up year
        cancer_surv_arr = np.zeros((END_AGE - START_AGE + 1, 10, 2))

        for i in range(10):
            cancer_surv_arr[:,i,0] = 1-(1-SURV[:,0])**(i+1)  # Cancer death
            cancer_surv_arr[:,i,1] = 1-(1-SURV[:,1])**(i+1)  # Other death
            # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year
    else: # multiple cancer sites
        cancer_surv_arr_lst = []
        for i in range(len(CANCER_SITES)):
            temp = SURV[SURV['Site']==CANCER_SITES[i]]
            temp = temp[['Cancer_Death','Other_Death']].to_numpy()  # 10 year survival
            temp = 1 - temp**(1/10)  # Converting to annual probability of death (assuming constant rate)

            # Converting into probability of death at each follow up year
            cancer_surv_arr = np.zeros((END_AGE - START_AGE + 1, 10, 2))

            for i in range(10):
                cancer_surv_arr[:,i,0] = 1-(1-temp[:,0])**(i+1)  # Cancer death
                cancer_surv_arr[:,i,1] = 1-(1-temp[:,1])**(i+1)  # Other death
                # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year

            cancer_surv_arr_lst.append(cancer_surv_arr)

    if len(CANCER_SITES) == 1:
        return ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC
    else:
        return ac_cdf, min_age, max_age, CANCER_PDF_lst, cancer_surv_arr_lst, CANCER_INC_lst
    # If we are running the model for multiple cancers, each cancer is a separate element in a list
