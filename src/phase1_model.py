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

def run_patient(pid, cancer_pdf, p_cancer):
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
    acMort_age = get_ac_mort_age(c.START_AGE, randVal)

    # Determine if a patient should develop cancer
    randVal = np.random.rand()
    if randVal < p_cancer: # patient develops cancer
        # Determine age of cancer
        randVal = np.random.rand() # TODO: same random value for all-cause mortality? or different one?
        cancer_age = get_all_cancer_age(c.START_AGE, cancer_pdf, randVal)

    # Determine if all-cause mortality or cancer happens first if patient gets cancer
    if cancer_age != -1 and (cancer_age > acMort_age): # cancer diagnosed after all-cause mortality
        cancer_age = -1 # patient doesn't end up getting cancer

    return cancer_age, acMort_age

def run_des(num_patients, cancer_pdf, p_cancer):
    """
    Runs discrete event simulation for total number of specified patients
    """
    cancerArr = np.zeros(len(range(c.START_AGE, c.END_AGE)))
    acMortArr = np.zeros(len(range(c.START_AGE, c.END_AGE)))

    for patient in range(num_patients):
        cancer_age, acMort_age = run_patient(patient, cancer_pdf, p_cancer)
        if cancer_age != -1:
            cancerArr[c.ALL_AGES.index(cancer_age)] += 1
        if acMort_age != -1:
            acMortArr[c.ALL_AGES.index(acMort_age)] += 1
    
    return cancerArr, acMortArr