# Configurations of the multi-cancer model
import numpy as np
import pandas as pd

# Aesthetic Preferences
np.set_printoptions(precision=5, suppress=True)


MODE = 'visualize'
# Options:
# - calibrate: run simulated annealing for cancer incidence
# - visualize: plot incidence and mortality

# Define cohort characteristics
COHORT_YEAR = 1950 # birth year of the cohort
COHORT_TYPE = 'am'  # am/af, All Male/All Female
START_AGE = 0
END_AGE = 100
NUM_PATIENTS = 100_000

# Define simulated annealing parameters
NUM_ITERATIONS = 100
STEP_SIZE = 0.001
VERBOSE = True
MASK_SIZE = 0.1  # value between 0 and 1

# Define input and output paths
INPUT_PATHS = {
    'life_tables': './data/life_tables/ssa_ac_mort.xlsx',
    'cancer_incid': './data/cancer_incidence/'
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

# Load all cancer incidence target data
CANCER_INC = pd.read_csv(INPUT_PATHS['cancer_incid'] + '1950_BC_All_Incidence.csv', index_col = 1)['Rate'].to_numpy()

# Loading in cancer pdf
CANCER_PDF = np.random.uniform(low=0.01, high=0.1, size=(END_AGE - START_AGE + 1))  # random to start

