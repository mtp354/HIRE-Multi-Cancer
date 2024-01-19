# Defines all functions used to run phase1 of the model (simplest version)
import configs as c
import pandas as pd
import numpy as np

def get_ac_mort_age(curr_age, randVal):
    """
    Outputs the age of all-cause mortality based on the current age
    and a random value between 0 and 1
    """
    # Get the conditional PDF
    condProb = c.pdf[curr_age:]
    condPDF = condProb / sum(condProb) # should sum to 1
    # Get conditional CDF
    condCDF = np.cumsum(condPDF)

    # Determine age of death
    return np.searchsorted(condCDF, np.random.rand()) + curr_age

def get_all_cancer_age(curr_age, cancer_pdf, randVal):
    """
    Outputs the age of being diagnosed with any cancer based on the current age
    and a random value between 0 and 1
    """
    # Generate CDF
    cdf = np.cumsum(cancer_pdf)

    # Determine age of cancer
    return np.searchsorted(cdf, np.random.rand()) + curr_age

def run_patient(pid, cancer_pdf):
    """
    Runs discrete event simulation for one patient
    :param pid: ID number of the individual
    Returns cancer age and all-cause mortality age (-1 means that did not die from the cause)
    """
    # Initialize age of all-cause mortality and all cancer
    acMort_age = -1
    cancer_age = -1

    # Determine age of all-cause mortality
    randVal = np.random.rand()
    new_acMort_age = get_ac_mort_age(c.START_AGE, randVal)
    
    # Determine age of any cancer
    randVal = np.random.rand() # TODO: same random value for all-cause mortality? or different one?
    new_cancer_age = get_all_cancer_age(c.START_AGE, cancer_pdf, randVal)

    # Determine if all-cause mortality or cancer happens first
    # If cancer happens first, update cancer_age
    if new_cancer_age <= new_acMort_age: # TODO: less than or less than or equal to?
        # Once patient gets cancer, they are pulled out of the model, they have no all-cause mortality age
        cancer_age = new_cancer_age
    else: # Patient does not get cancer and dies from all-cause mortality
        acMort_age = new_acMort_age

    return cancer_age, acMort_age

def run_des(num_patients, cancer_pdf):
    """
    Runs discrete event simulation for total number of specified patients
    """
    cancerArr = np.zeros(len(range(c.START_AGE, c.END_AGE)))
    acMortArr = np.zeros(len(range(c.START_AGE, c.END_AGE)))

    for patient in range(num_patients):
        cancer_age, acMort_age = run_patient(patient, cancer_pdf)
        if cancer_age != -1:
            cancerArr[c.ALL_AGES.index(cancer_age)] += 1
        if acMort_age != -1:
            acMortArr[c.ALL_AGES.index(acMort_age)] += 1
    
    # Get number of patients who are alive at the start of each year
    live = np.zeros(len(range(c.START_AGE, c.END_AGE)))
    for i in range(len(live)):
        if i == 0:
            live[i] = c.NUM_PATIENTS - acMortArr[i]
        else:
            live[i] = live[i - 1] - acMortArr[i]
    
    # Get cancer incidence for each age
    cancer_incid = cancerArr / live * 100_000
    # Only from ages 25 - 70 to align with target data
    cancer_incid = cancer_incid[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]
    return cancer_incid


    


   