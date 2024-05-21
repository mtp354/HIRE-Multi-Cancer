# A super simple discrete event simulation model based on the simpy package.
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import configs as c
from classes import *
from tqdm import tqdm

if __name__ == '__main__':
    start = timer()
    if c.MODE == 'visualize':
        # Run simplest verion of the model
        # Initialize cohort-specific parameters
        ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC = c.select_cohort(c.COHORT_YEAR, c.COHORT_SEX, c.COHORT_RACE)
        model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr)
        print(objective(model.run(CANCER_PDF).cancerIncArr, min_age, max_age, CANCER_INC))
        plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(CANCER_PDF).cancerIncArr[:-1], label='Model', color='blue')
        plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
        plt.show()
    elif c.MODE == 'calibrate':
        if c.MULTI_COHORT_CALIBRATION:
            print(f"RUNNING MULTI-COHORT CALIBRATION: FIRST COHORT = {c.FIRST_COHORT}, LAST COHORT = {c.LAST_COHORT}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
            print(f"CANCERS = {c.CANCER_SITES}")
            if c.FIRST_COHORT < c.LAST_COHORT: # calibrate in ascending birth years
                for cohort in tqdm(range(c.FIRST_COHORT, c.LAST_COHORT + 1)):
                    # Initialize cohort-specific parameters
                    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC = c.select_cohort(cohort, c.COHORT_SEX, c.COHORT_RACE)
                    model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr)
                    # Run calibration for simplest verion of the model
                    best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age)
                    # Save as numpy file, time_stamped
                    if c.SAVE_RESULTS:
                        np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
                    plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
                    plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
                    plt.legend(loc='upper left')
                    plt.xlabel('Age')
                    plt.ylabel('Incidence (per 100k)')
                    plt.title(f"Cancer Incidence by Age for Birthyear={cohort}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
                    plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
                    plt.clf()
            else: # calibrate in descending birth years
                for cohort in tqdm(range(c.FIRST_COHORT, c.LAST_COHORT + 1, -1)):
                    # Initialize cohort-specific parameters
                    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC = c.select_cohort(cohort, c.COHORT_SEX, c.COHORT_RACE)
                    model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr)
                    # Run calibration for simplest verion of the model
                    best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age)
                    # Save as numpy file, time_stamped
                    if c.SAVE_RESULTS:
                        np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
                    plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
                    plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
                    plt.legend(loc='upper left')
                    plt.xlabel('Age')
                    plt.ylabel('Incidence (per 100k)')
                    plt.title(f"Cancer Incidence by Age for Birthyear={cohort}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
                    plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
                    plt.clf()
        else:
            print(f"RUNNING CALIBRATION: COHORT = {c.COHORT_YEAR}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
            print(f"CANCERS = {c.CANCER_SITES}")
            # Initialize cohort-specific parameters
            ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC = c.select_cohort(c.COHORT_YEAR, c.COHORT_SEX, c.COHORT_RACE)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr)
            # Run calibration for simplest verion of the model
            best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age)
            # Save as numpy file, time_stamped
            if c.SAVE_RESULTS:
                np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
            plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
            plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
            plt.legend(loc='upper left')
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
            plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
            plt.clf()

    end = timer()
    print(f'total time: {timedelta(seconds=end-start)}')



