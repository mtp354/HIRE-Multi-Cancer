# Defines all functions used to run phase1 of the model (simplest version)
import configs as c
import pandas as pd
import numpy as np

class Patient:
    def __init__(self, pid, starting_age=0):
        """
        Initializes the object with the given `pid`, `age`, `cancer_pdf`, and `randVal`.

        :param pid: The patient ID.
        :param age: The age of the patient.
        :param cancer_pdf: The PDF file containing information about cancer.
        :param randVal: The random value for initialization.

        :return: None
        """
        self.pid = pid
        self.age = starting_age
        self.karma = np.random.rand()
        self.history = [(self.age, 0)]
    
    def run(self, cancer_pdf):
        """
        Runs the discrete event simulation for one patient
        """
        condProb = c.pdf[self.age:]  # Get the conditional PDF
        condCDF = np.cumsum(condProb / sum(condProb))  # Get conditional CDF
        ac_age = np.searchsorted(condCDF, self.karma) + self.age  # Determine age of death

        cancer_cdf = np.cumsum(cancer_pdf)  # Generate CDF
        cancer_age = np.searchsorted(cancer_cdf, self.karma) + self.age  # Determine age of cancer
        if cancer_age <= ac_age:  # If cancer happens before death
            self.history.append((cancer_age, 1))  # Add to history

        self.history.append((ac_age, 2))

        return self.history




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


test_cancer_pdf = np.random.rand(100)
james = Patient(0)
james.run()

