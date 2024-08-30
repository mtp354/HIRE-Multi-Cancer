# A super simple discrete event simulation model based on the simpy package.
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import src.configs as c
from src.classes import *
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pickle

def run_model(cancer_sites: [""], cohort: int, sex: str, race: str, start_age = 0, end_age = 100, save=True):
    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, CANCER_INC = (
        c.select_cohort_app(cancer_sites, cohort, sex, race[3:], start_age, end_age)
    )
    model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, len(cancer_sites))
    model.run(CANCER_PDF)
    inc_df = pd.DataFrame(model.cancerIncArr, columns=["Incidence per 100k"])
    inc_df = inc_df.reset_index().rename(columns={'index': 'Age'})

    # Output model incidence
    if save:
        app_dir = Path(__file__).parent / "app/data/incidence"
        inc_df.to_csv(f"{str(app_dir)}/{cohort}_{sex}_{race}_{cancer_sites[0]}.xlsx")
    return inc_df


def run_calibration(cohort):
    # Initialize cohort-specific parameters
    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, CANCER_INC = c.select_cohort(cohort, c.COHORT_SEX, c.COHORT_RACE)
    model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, len(c.CANCER_SITES))

    # Run calibration 
    best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age)
    # Save as numpy file
    if c.SAVE_RESULTS:
        np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
    try:
        plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
        plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.ylim(0, CANCER_INC.max() + 30)
        plt.title(f"Cancer Incidence by Age for Birthyear={cohort}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
        plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{cohort}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
        plt.clf()
    except Exception as e:
        print(f"An error occurred while plotting or saving the figure: {e}")
if __name__ == '__main__':
    start = timer()
    if c.MODE == 'visualize':
        # Run simplest verion of the model: creates plot of model incidence vs SEER incidence
        # Runs for a single cancer site or multiple cancer sites
        print(f"RUNNING MODEL VISUALIZATION: COHORT = {c.COHORT_YEAR}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
        print(f"CANCERS = {c.CANCER_SITES}")
        # Initialize cohort-specific parameters
        if len(c.CANCER_SITES) == 1: # single cancer site
            ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, CANCER_INC = c.select_cohort(c.COHORT_YEAR, c.COHORT_SEX, c.COHORT_RACE)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, len(c.CANCER_SITES))
            print(objective(model.run(CANCER_PDF).cancerIncArr, min_age, max_age, CANCER_INC))
        
            # Output model incidence, cancer count, alive count
            df = pd.DataFrame(model.cancerIncArr, columns = ['Incidence'])
            df['Cancer_Count'] = model.cancerCountArr
            df['Alive_Count'] = model.aliveCountArr
            df.to_excel(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}_SUMMARY.xlsx")
            
            if c.SOJOURN_TIME:
                with open(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}_LOG_sj.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)
            else:
                with open(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}_LOG_nh.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)

            # Limit the plot's y-axis to just above the highest SEER incidence
            plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(CANCER_PDF).cancerIncArr[:-1], label='Model', color='blue')
            plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
            plt.legend(loc='upper left')
            plt.ylim(0, CANCER_INC.max() + 30)
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
            plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
            plt.clf()
            plt.show()
        
        else: # multiple cancer sites
            ac_cdf, min_age, max_age, CANCER_PDF_lst, cancer_surv_arr_lst, cancer_surv_arr_lst_ed, CANCER_INC_lst = c.select_cohort(c.COHORT_YEAR,
                                                                                                            c.COHORT_SEX, c.COHORT_RACE)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr_lst, cancer_surv_arr_lst_ed, len(c.CANCER_SITES))
            plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(CANCER_PDF_lst).cancerIncArr[:-1], label='Model', color='blue')
            
            # Output model incidence, cancer count, alive count
            df = pd.DataFrame(model.cancerIncArr, columns = ['Incidence'])
            df['Cancer_Count'] = model.cancerCountArr
            df['Alive_Count'] = model.aliveCountArr
            df.to_excel(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{str(c.CANCER_SITES)}_SUMMARY.xlsx")
            
            if c.SOJOURN_TIME:
                cancer_sites_str = '_'.join(c.CANCER_SITES_ED)               
                with open(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{cancer_sites_str}_LOG_sj.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)
            else:
                cancer_sites_str = '_'.join(c.CANCER_SITES)
                with open(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{cancer_sites_str}_LOG_nh.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)

            plt.legend(loc='upper left')
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={str(c.CANCER_SITES)}")
            plt.show()

    elif c.MODE == 'cancer_dist': # Saves a plot of the calibrated cancer cdf and pdf
        # Initialize cohort-specific parameters
        ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, CANCER_INC = c.select_cohort(c.COHORT_YEAR, c.COHORT_SEX, c.COHORT_RACE)
        # Plot pdf
        fig = plt.figure(figsize = (10,5))
        plt.bar(range(c.START_AGE, c.END_AGE+1), CANCER_PDF) # doesn't start at age 0, starts at age 1
        plt.xlabel("Age")
        plt.ylabel("PDF")
        plt.savefig(c.PATHS['plots'] + f"cancer_pdf_{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}.png")
        plt.clf()

        fig = plt.figure(figsize = (10,5))
        plt.bar(range(c.START_AGE, c.END_AGE+1), np.cumsum(CANCER_PDF)) # doesn't start at age 0, starts at age 1
        plt.xlabel("Age")
        plt.ylabel("CDF")
        plt.savefig(c.PATHS['plots'] + f"cancer_cdf_{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}.png")
        plt.clf()

    elif c.MODE == 'calibrate':
        if c.MULTI_COHORT_CALIBRATION:
            print(f"RUNNING MULTI-COHORT CALIBRATION: FIRST COHORT = {c.FIRST_COHORT}, LAST COHORT = {c.LAST_COHORT}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
            print(f"CANCERS = {c.CANCER_SITES}")
            with mp.Pool(processes=c.NUM_PROCESSES) as pool:
                results = list(tqdm(pool.imap(run_calibration, range(c.FIRST_COHORT, c.LAST_COHORT + 1)), total=c.LAST_COHORT - c.FIRST_COHORT + 1))

        else:
            print(f"RUNNING CALIBRATION: COHORT = {c.COHORT_YEAR}, SEX = {c.COHORT_SEX}, RACE = {c.COHORT_RACE}")
            print(f"CANCERS = {c.CANCER_SITES}")
            # Initialize cohort-specific parameters
            ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, CANCER_INC = c.select_cohort(c.COHORT_YEAR, c.COHORT_SEX, c.COHORT_RACE)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, len(c.CANCER_SITES))
            # Run calibration for simplest verion of the model
            best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age)
            # Save as numpy file, time_stamped
            if c.SAVE_RESULTS:
                np.save(c.PATHS['calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
            # Limit the plot's y-axis to just above the highest SEER incidence
            plt.plot(np.arange(c.START_AGE, c.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
            plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
            plt.legend(loc='upper left')
            plt.ylim(0, CANCER_INC.max() + 30)
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Sex={c.COHORT_SEX}, Race={c.COHORT_RACE}, Site={c.CANCER_SITES[0]}")
            plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.COHORT_YEAR}_{c.CANCER_SITES[0]}.png", bbox_inches='tight')
            plt.clf()

    end = timer()
    print(f'total time: {timedelta(seconds=end-start)}')



