# A super simple discrete event simulation model based on the simpy package.

import configs as c
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta
import phase1_model as phase1
import matplotlib.pyplot as plt
import phase1_calibration as phase1_calib

if __name__ == '__main__':
    if c.MODE == 'phase1':
        # Run simplest verion of the model
        start = timer()

        cancerArr, acMortArr = phase1.run_des(c.NUM_PATIENTS)
        cancerIncid = phase1.get_des_outputs(cancerArr, acMortArr)

        # acMort_age_output = np.zeros(101)
        # for pid in range(c.NUM_PATIENTS): # Iterate over total number of patients
        #     acMort_age = phase1.run_des(pid) # Run one patient
        #     acMort_age_output[acMort_age] += 1
        
        # # Plot distribution of all-cause mortality ages
        # fig = plt.figure(figsize = (10,5))
        # plt.bar(range(101), age_output)
        # plt.xlabel("Age")
        # plt.ylabel("Number of Patients")
        # plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'des_acmort_age_dist.png')

        end = timer()
        print(f'total time: {timedelta(seconds=end-start)}')

    elif c.MODE == 'phase1_calib':
        # Run calibration for simplest verion of the model
        start = timer()

        # Generate random starting cancer pdf
        init_cancer_pdf = np.random.rand(len(range(25,71)))
        init_cancer_pdf /= init_cancer_pdf.sum()

        final_cancer_pdf = phase1_calib.anneal(init_cancer_pdf)

        # Save as numpy file
        np.save(c.OUTPUT_PATHS['calibration'] + 'cancer_pdf_01192024', final_cancer_pdf)

        end = timer()
        print(f'total time: {timedelta(seconds=end-start)}')