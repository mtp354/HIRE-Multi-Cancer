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
COHORT_SEX = 'Female'  # Female/Male
COHORT_RACE = 'White'  # Black/White
NUM_PATIENTS = 100_000
CANCER_SITES = ['Breast']
# Full list:
# MP 'Bladder' 'Breast' 'Cervical' 'Colorectal' 'Esophageal' 
# JP 'Gastric' 'Lung' 'Prostate' 'Uterine'
# FL 'Pancreatic' 'Ovarian' 'Kidney' 'Brain' 'Liver' 'Gallbladder'

# Define simulated annealing parameters
NUM_ITERATIONS = 1_000
START_TEMP = 10
STEP_SIZE = 0.001
VERBOSE = True
MASK_SIZE = 0.1  # value between 0 and 1, the fraction of values to modify each step
LOAD_LATEST = True  # If true, load the latest cancer_pdf from file as starting point

# Define input and output paths
PATHS = {
    'incidence': './data/cancer_incidence/',
    'mortality': './data/mortality/',
    'survival': './data/cancer_survival/',
    'calibration': './outputs/calibration/',
    'plots': './outputs/plots/'
}

# Load input data
CANCER_INC = pd.read_csv(f'{PATHS["incidence"]}Incidence.csv')
CANCER_INC = CANCER_INC[CANCER_INC['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest


# Load in mortality data
MORT = pd.read_csv(f'{PATHS["mortality"]}Mortality.csv')
MORT = MORT[~MORT['Site'].isin(CANCER_SITES)]  # Removing the cancers of interest
MORT = MORT.groupby(['Cohort','Age','Sex','Race']).agg({'Rate':'sum'}).reset_index()  # Summing over the remaining sites

# Load in Survival data
SURV = pd.read_csv(f'{PATHS["survival"]}Survival.csv')  # This is the 10 year survival by cause
SURV = SURV[SURV['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest

# Selecting Cohort
CANCER_INC.query('Sex == @COHORT_SEX & Race == @COHORT_RACE & Cohort == @COHORT_YEAR', inplace=True)
MORT.query('Sex == @COHORT_SEX & Race == @COHORT_RACE & Cohort == @COHORT_YEAR', inplace=True)
SURV.query('Sex == @COHORT_SEX & Race == @COHORT_RACE', inplace=True)

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
min_age = max(1975 - COHORT_YEAR, 0)
max_age = min(2018 - COHORT_YEAR, 84)

# Loading in cancer pdf, this is the thing that will be optimized over
CANCER_PDF = 0.002 * np.ones(END_AGE - START_AGE + 1)  # starting from 0 incidence and using bias optimization
CANCER_PDF[:35] = 0.0
if LOAD_LATEST:
    list_of_files = glob.glob(f'{PATHS["calibration"]}*')
    latest_file = max(list_of_files, key=os.path.getctime)
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
