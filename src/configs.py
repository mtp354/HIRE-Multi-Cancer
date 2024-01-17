# Configurations of the multi-cancer model
import numpy as np
import pandas as pd

# Options:
# - phase1: annual risk of getting all cancers, all-cause mortality
MODE = 'phase1'

# Define cohort characteristics
COHORT_YEAR = 1950 # birth year of the cohort
COHORT_TYPE = 'am' # options: am (all men), af (all women)
START_AGE = 18
END_AGE = 100 # Everyone dies before age 100 TODO: Chin said AGE 79 this was the last year of SEER 2019? Is this right?
NUM_PATIENTS = 1_000_000

# Cycle length
NUM_CYCLES = 1 # number of cycles per year
CYCLE_LENGTH = 1 / NUM_CYCLES

# Define input and output paths
INPUT_PATHS = {
    'life_tables': '../data/life_tables/ssa_ac_mort.xlsx'
}

OUTPUT_PATHS = {
    'ac_mort_plots': '../outputs/plots/ac_mort/'
}

# Import life tables: age from 0 too 100 (assumes everyone dies at age 100)
life_table = pd.read_excel("../data/life_tables/ssa_ac_mort.xlsx", sheet_name=COHORT_TYPE, index_col=0)
life_table = life_table.iloc[:,0:4]
# Create the cdf and pdf for all-cause mortality
# Get proportion of people who died at start of the year -> cumulative distribution function (CDF)
cdf = 1 - life_table['Start_Num']/100000
# Get the probability distribution function (PDF) = CDF(n) - CDF(n-1)
pdf = np.diff(np.array(cdf)) # sums to 1