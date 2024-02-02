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
COHORT_YEAR = 1950 # birth year of the cohort
COHORT_TYPE = 'am'  # am/af, All Male/All Female
START_AGE = 0
END_AGE = 100
NUM_PATIENTS = 100_000

# Define simulated annealing parameters
NUM_ITERATIONS = 1_000
START_TEMP = 1
STEP_SIZE = 0.0001
VERBOSE = True
MASK_SIZE = 0.1  # value between 0 and 1, the fraction of values to modify each step
LOAD_LATEST = True  # If true, load the latest cancer_pdf from file as starting point

# Define input and output paths
INPUT_PATHS = {
    'life_tables': './data/life_tables/ssa_ac_mort.xlsx',
    'cancer_incid': './data/cancer_incidence/',
    'cancer_surv': './data/cancer_survival/Multi_Cancer_Survival.xlsx',
}

OUTPUT_PATHS = {
    'ac_mort_plots': './outputs/plots/ac_mort/',
    'calibration': './outputs/calibration/'
}

# Import life tables: age from 0 too 100 (assumes everyone dies at age 100)
life_table = pd.read_excel(INPUT_PATHS['life_tables'], sheet_name=COHORT_TYPE, index_col=0)
life_table = life_table.iloc[:,0:4]
# Create the cdf and pdf for all-cause mortality
# Get proportion of people who died at start of the year -> cumulative distribution function (CDF)
ac_cdf = 1 - life_table['Start_Num']/100000
# Get the probability distribution function (PDF) = CDF(n) - CDF(n-1)
ac_pdf = np.diff(np.array(ac_cdf)) # sums to 1
condProb = ac_pdf[START_AGE:]  # Get the conditional PDF
condCDF = np.cumsum(condProb / sum(condProb))  # Get conditional CDF

# Load all cancer incidence target data
CANCER_INC = pd.read_csv(INPUT_PATHS['cancer_incid'] + '1950_BC_All_Incidence.csv', index_col = 1)['Rate'].to_numpy()

# Loading in cancer pdf, 
CANCER_PDF = np.zeros(END_AGE - START_AGE + 1)  # starting from 0 incidence and using bias optimization
if LOAD_LATEST:
    list_of_files = glob.glob('./outputs/calibration/*')
    latest_file = max(list_of_files, key=os.path.getctime)
    CANCER_PDF = np.load(latest_file)



# Loading in cancer survival data
cancer_surv = pd.read_excel(INPUT_PATHS['cancer_surv'], sheet_name=COHORT_TYPE)
cancer_surv = cancer_surv[cancer_surv['Birth_Year'] == COHORT_YEAR].iloc[:,1:].to_numpy()  # Selecting only chosen cohort
age_min, age_max = int(cancer_surv[:,0].min()), int(cancer_surv[:,0].max())
# Converting to numpy array and padding with 0 for missing data
cancer_surv_arr = np.zeros((END_AGE - START_AGE + 1, 10, 2))  # Age, Years after Diagnosis, Cause of Death
cancer_surv = cancer_surv[:,2:].reshape(age_max-age_min+1, 10, 2)
cancer_surv_arr[age_min:age_max+1, :, :] = cancer_surv

