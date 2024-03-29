# Configurations of the multi-cancer model
import numpy as np
import pandas as pd
import glob
import os

# Aesthetic Preferences
np.set_printoptions(precision=5, suppress=True)

MODE = 'visualize'
# Options:
# - calibrate: run simulated annealing for cancer incidence
# - visualize: plot incidence and mortality
SAVE_RESULTS = True  # whether to save results to file

# Define cohort characteristics
COHORT_YEAR = 1920 # birth year of the cohort
COHORT_SEX = 'Female'  # Female/Male
COHORT_RACE = 'White'  # Black/White
NUM_PATIENTS = 100_000
CANCER_SITES = ['Breast']
# Full list:
# 'Bladder' 'Breast' 'Cervical' 'Colorectal' 'Esophageal' 'Gastric' 'Lung' 'Prostate' 'Uterine'
# 'Pancreatic' 'Ovarian' 'Kidney' 'Brain' 'Liver' 'Gallbladder'

# Define simulated annealing parameters
NUM_ITERATIONS = 1_000
START_TEMP = 10
STEP_SIZE = 0.00001
VERBOSE = True
MASK_SIZE = 0.1  # value between 0 and 1, the fraction of values to modify each step
LOAD_LATEST = False  # If true, load the latest cancer_pdf from file as starting point


# Define input and output paths
INPUT_PATHS = {
    'data': './data/',
    'cancer_surv': './data/cancer_survival/Multi_Cancer_Survival.xlsx'
}

OUTPUT_PATHS = {
    'calibration': './outputs/calibration/'
}

# Load input data
CANCER_INC = pd.read_csv('./data/Incidence.csv')
CANCER_INC = CANCER_INC[CANCER_INC['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest

# Load in mortality data
MORT = pd.read_csv('./data/Mortality.csv')
MORT = MORT[~MORT['Site'].isin(CANCER_SITES)]  # Removing the cancers of interest
MORT = MORT.groupby(['Cohort','Age','Sex','Race']).agg({'Rate':'sum'}).reset_index()  # Summing over the remaining sites

# Load in Survival data
SURV = pd.read_csv('./data/10_Yr_Survival.csv')
SURV = SURV[SURV['Site'].isin(CANCER_SITES)]  # keeping the cancers of interest

# Selecting Cohort
CANCER_INC.query('Sex == @COHORT_SEX & Race == @COHORT_RACE & Cohort == @COHORT_YEAR', inplace=True)
MORT.query('Sex == @COHORT_SEX & Race == @COHORT_RACE & Cohort == @COHORT_YEAR', inplace=True)
SURV.query('Sex == @COHORT_SEX & Race == @COHORT_RACE & Cohort == @COHORT_YEAR', inplace=True)


# Only running for the years we have available data
START_AGE = max(MORT['Age'].min(), CANCER_INC['Age'].min(), 35)
END_AGE = min(MORT['Age'].max(), CANCER_INC['Age'].max())

# Create the cdf all-cause mortality
MORT = MORT[(MORT['Age'] >= START_AGE) & (MORT['Age'] <= END_AGE)]
ac_cdf = np.cumsum(MORT['Rate'].to_numpy())/100000
ac_pdf = np.diff(ac_cdf)

# Load all cancer incidence target data
CANCER_INC = CANCER_INC[(CANCER_INC['Age'] >= START_AGE) & (CANCER_INC['Age'] <= END_AGE)]
CANCER_INC = CANCER_INC['Rate'].to_numpy()

# Loading in cancer pdf, 
CANCER_PDF = 0.001 * np.ones(END_AGE - START_AGE + 1)  # starting from 0 incidence and using bias optimization
if LOAD_LATEST:
    list_of_files = glob.glob('./outputs/calibration/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    CANCER_PDF = np.load(latest_file)

# Loading in cancer survival data
# SURV = SURV[['Cancer_Death','Other_Death']].to_numpy()[START_AGE:END_AGE+1,:] # 10 year survival

# Loading in cancer survival data
cancer_surv_arr = pd.read_excel(INPUT_PATHS['cancer_surv'], sheet_name='am')
cancer_surv_arr = cancer_surv_arr[cancer_surv_arr['Birth_Year'] == COHORT_YEAR]  # Selecting only chosen cohort

cancer_surv_arr = cancer_surv_arr[(cancer_surv_arr['Age'] >= START_AGE) & (cancer_surv_arr['Age'] <= END_AGE)]
cancer_surv_arr = cancer_surv_arr[['Cancer_Death','Other_Death']].to_numpy().reshape((END_AGE - START_AGE + 1, 10, 2))
cancer_surv_arr[np.isnan(cancer_surv_arr)] = 0  # Converting to numpy array and padding with 0 for missing data


