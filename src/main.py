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

        # Define cancer probability and cancer pdf
        p_cancer_df = pd.read_excel(c.OUTPUT_PATHS['calibration'] + 'p_cancer_01252024.xlsx', index_col=0)
        p_cancer = p_cancer_df.iloc[0,0]
        cancer_pdf = np.load(c.OUTPUT_PATHS['calibration'] + 'cancer_pdf_01252024.npy')
        
        # Run model
        cancerArr, acMortArr = phase1.run_des(c.NUM_PATIENTS, cancer_pdf, p_cancer)
        # Save number of patients who develop cancer at each age or die from all-cause mort at each age as dataframes
        cancer_df = pd.DataFrame(data = cancerArr, index = range(c.START_AGE, c.END_AGE), columns = ['Num_Patients'])
        acMort_df = pd.DataFrame(data = acMortArr, index = range(c.START_AGE, c.END_AGE), columns = ['Num_Patients'])
        cancer_df.to_excel(c.OUTPUT_PATHS['phase1'] + 'cancer_counts.xlsx')
        acMort_df.to_excel(c.OUTPUT_PATHS['phase1'] + 'acMort_counts.xlsx')

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
        print("RUNNING CALIBRATION")
        # Initial cancer probability value
        p_cancer = 0.02679

        # Generate random starting cancer pdf
        init_cancer_pdf = np.random.rand(len(range(25,71)))
        init_cancer_pdf /= init_cancer_pdf.sum()

        # # Import previous calibration as starting pdf
        # init_cancer_pdf = np.load(c.OUTPUT_PATHS['calibration'] + 'cancer_pdf_01192024_4.npy')

        # Run calibration
        final_cancer_pdf, final_p_cancer = phase1_calib.anneal(init_cancer_pdf, p_cancer)
        final_p_cancer_df = pd.DataFrame(np.array([final_p_cancer]), columns = ['prob'])

        # Save as numpy file
        np.save(c.OUTPUT_PATHS['calibration'] + 'cancer_pdf_01252024', final_cancer_pdf)
        final_p_cancer_df.to_excel(c.OUTPUT_PATHS['calibration'] + 'p_cancer_01252024.xlsx')

        end = timer()
        print(f'total time: {timedelta(seconds=end-start)}')