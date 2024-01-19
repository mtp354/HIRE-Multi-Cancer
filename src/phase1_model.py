# Defines all functions used to run phase1 of the model (simplest version)
import simpy
import random
import configs as c
import pandas as pd
import numpy as np

def get_ac_mort_age(curr_age, randVal):
    """Outputs the age of all-cause mortality based on the current age
       and a random value between 0 and 1"""
    # Get the conditional PDF
    condProb = c.pdf[curr_age:]
    condPDF = condProb / sum(condProb) # should sum to 1
    # Get conditional CDF
    condCDF = np.cumsum(condPDF)

    # Determine the age of death
    return np.searchsorted(condCDF, np.random.rand())

def run_des(pid):
    """
    Runs discrete event simulation for one patient
    :param pid: ID number of the individual
    """
    # Determine age of all-cause mortality
    randVal = np.random.rand()
    acMort_age = get_ac_mort_age(c.START_AGE, randVal)
    return acMort_age
    


   