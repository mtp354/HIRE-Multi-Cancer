# A super simple discrete event simulation model based on the simpy package.
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import configs as c
from classes import *

if __name__ == '__main__':
    start = timer()
    model = DiscreteEventSimulation()
    if c.MODE == 'visualize':
        # Run simplest verion of the model
        print(objective(model.run(c.CANCER_PDF).cancerIncArr, c.CANCER_INC))
        plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(c.CANCER_PDF).cancerIncArr[:-1], label='Model', color='blue')
        plt.plot(np.arange(c.min_age, c.max_age+1), c.CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
        plt.show()
    elif c.MODE == 'calibrate':
        print(f"RUNNING CALIBRATION: COHORT = {c.COHORT_YEAR}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
        print(f"CANCERS = {c.CANCER_SITES}")
        # Run calibration for simplest verion of the model
        # c.CANCER_PDF[:30] = 0
        # c.CANCER_PDF = savgol_filter(c.CANCER_PDF, 40, 4, mode='interp')  # smoothing
        best = simulated_annealing(model)
        # Save as numpy file, time_stamped
        if c.SAVE_RESULTS:
            np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
        plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
        plt.plot(np.arange(c.min_age, c.max_age+1), c.CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
        plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')

    end = timer()
    print(f'total time: {timedelta(seconds=end-start)}')



