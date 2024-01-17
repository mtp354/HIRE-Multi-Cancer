# A super simple discrete event simulation model based on the simpy package.

import configs as c
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import phase1_model as phase1
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if c.MODE == 'phase1':
        # Run simplest verion of the model
        start = timer()

        age_output = np.zeros(101)
        for pid in range(c.NUM_PATIENTS): # Iterate over total number of patients
            acMort_age = phase1.run_des(pid) # Run one patient
            age_output[acMort_age] += 1
        
        # Plot distribution of all-cause mortality ages
        fig = plt.figure(figsize = (10,5))
        plt.bar(range(101), age_output)
        plt.xlabel("Age")
        plt.ylabel("Number of Patients")
        plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'des_acmort_age_dist.png')

        end = timer()
        print(f'total time: {timedelta(seconds=end-start)}')