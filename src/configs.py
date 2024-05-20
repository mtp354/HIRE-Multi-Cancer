# Configurations of the multi-cancer model
import numpy as np
import pandas as pd
import glob
import os

# Aesthetic Preferences
np.set_printoptions(precision=5, suppress=True)

MODE = 'calibrate'
# Options:
# - calibrate: run simulated annealing for cancer incidence
# - visualize: plot incidence and mortality
SAVE_RESULTS = True  # whether to save results to file

# Define cohort characteristics
COHORT_YEAR = 1940  # birth year of the cohort
START_AGE = 0
END_AGE = 100
COHORT_SEX = 'Male'  # Female/Male
COHORT_RACE = 'White'  # Black/White
NUM_PATIENTS = 100_000
CANCER_SITES = ['Pancreas']
# Full list:
# MP 'Bladder' 'Breast' 'Cervical' 'Colorectal' 'Esophageal' 
# JP 'Gastric' 'Lung' 'Prostate' 'Uterine'
# FL 'Pancreas' 'Ovarian' 'Kidney' 'Brain' 'Liver' 'Gallbladder'

# Raise exceptions for male/ovarian, male/uterine, male/cervical, female/prostrate
if COHORT_SEX == 'Male' and ('Ovarian' in CANCER_SITES or 'Uterine' in CANCER_SITES or 'Cervical' in CANCER_SITES):
    raise Exception("Cancer site and cohort sex combination is not valid in configs.py")
elif COHORT_SEX == 'Female' and 'Prostate' in CANCER_SITES:
    raise Exception("Cancer site and cohort sex combination is not valid in configs.py")

# Define simulated annealing parameters
NUM_ITERATIONS = 1_000
START_TEMP = 10
STEP_SIZE = 0.001
VERBOSE = True
MASK_SIZE = 0.1  # value between 0 and 1, the fraction of values to modify each step
LOAD_LATEST = False  # If true, load the latest cancer_pdf from file as starting point
MULTI_COHORT_CALIBRATION = False
# You can either do multi-calibration by increasing (FIRST_COHORT < LAST_COHORT)
# or decreasing cohorts (FIRST_COHORT > LAST_COHORT)
# But you must be multi-calibration in ascending order first before doing descending order
if MULTI_COHORT_CALIBRATION: 
    FIRST_COHORT = 1941
    LAST_COHORT = 1945

# Define input and output paths
PATHS = {
    'incidence': '../data/cancer_incidence/',
    'mortality': '../data/mortality/',
    'survival': '../data/cancer_survival/',
    'calibration': '../outputs/calibration/',
    'plots_calibration': '../outputs/calibration/plots/',
    'sojourn_time': '../data/Sojourn Times/',
    'plots': '../outputs/plots/'
}

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

    # Load in sojorn times
    sojourn = pd.read_csv(PATHS['sojourn_time'] + 'Sojourn Estimates.csv')
    sojourn = sojourn[sojourn['Site'].isin(CANCER_SITES)]

    CANCER_INC.query('Sex == @sex & Race == @race & Cohort == @birthyear', inplace=True)
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
    CANCER_INC = CANCER_INC['Rate'].to_numpy()
    # For plotting and objective, we only compare years we have data
    min_age = max(1975 - birthyear, 0)
    max_age = min(2018 - birthyear, 84)

    # Loading in cancer pdf, this is the thing that will be optimized over
    CANCER_PDF = 0.002 * np.ones(END_AGE - START_AGE + 1)  # starting from 0 incidence and using bias optimization
    CANCER_PDF[:35] = 0.0
    if LOAD_LATEST:
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
        if FIRST_COHORT < LAST_COHORT: # ascending birth year calibration
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
        else: # descending birth year calibration
            # Get the min calibrated cohort year that is just above the COHORT_YEAR
            min_year = birthyear + 1
            final_list = []
            for file in list_of_files:
                if f'_{min_year}_' in file:
                    final_list.append(file)
        # Read the latest file
        latest_file = max(final_list, key=os.path.getctime)
        CANCER_PDF = np.load(latest_file)

    # Loading in cancer survival data
    SURV = SURV[['Cancer_Death','Other_Death']].to_numpy()  # 10 year survival
    SURV = 1 - SURV**(1/10)  # Converting to annual probability of death (assuming constant rate)

    # Converting into probability of death at each follow up year
    cancer_surv_arr = np.zeros((END_AGE - START_AGE + 1, 10, 2))

    for i in range(10):
        cancer_surv_arr[:,i,0] = 1-(1-SURV[:,0])**(i+1)  # Cancer death
        cancer_surv_arr[:,i,1] = 1-(1-SURV[:,1])**(i+1)  # Other death
        # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year

    return ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC
