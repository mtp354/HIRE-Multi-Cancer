# Calculates cancer incidence from outputs of the DES model
import configs as c
import pandas as pd
import numpy as np

# Load in model outputs of cancer and all-cause mortality
cancer_df = pd.read_excel(c.OUTPUT_PATHS['phase1'] + 'cancer_counts.xlsx', index_col= 0)
acMort_df = pd.read_excel(c.OUTPUT_PATHS['phase1'] + 'acMort_counts.xlsx', index_col= 0)
# Convert to numpy arrays
cancerArr = cancer_df['Num_Patients'].to_numpy()
acMortArr = acMort_df['Num_Patients'].to_numpy()

# Get number of patients who are alive at the start of each year
live = np.zeros(len(range(c.START_AGE, c.END_AGE)))
for i in range(len(live)):
    if i == 0:
        live[i] = c.NUM_PATIENTS - acMortArr[i]
    else:
        live[i] = live[i - 1] - acMortArr[i]
# Restrict live and cancer Arr to the cancer incidence ages
live = live[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]
cancerArr = cancerArr[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]

# Get cancer incidence for each age
cancer_incid = cancerArr / live * 100_000
# Only from ages 25 - 70 to align with target data
cancer_incid = cancer_incid[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]

# Output cancer incidence to Excel
cancer_incid_df = pd.DataFrame(data = cancer_incid, index = range(25, 71), columns = ['incidence'])
cancer_incid_df.to_excel(c.OUTPUT_PATHS['phase1'] + 'all_cancer_incidence.xlsx')

    


   