# A super simple discrete event simulation model based on the simpy package.
import numpy as np
from timeit import default_timer as timer
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import configs as c
from configs import Config
from classes import *
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import pickle
import argparse
from functools import partial

### Parse arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Configure the simulation parameters.')

parser.add_argument('--mode', type=str, default='visualize', help='Mode of operation')
parser.add_argument('--save_results', type=str2bool, default=True, help='Whether to save results')
parser.add_argument('--sojourn_time', type=str2bool, default=False, help='Whether to use sojourn time')
parser.add_argument('--gof_smoothing', type=str2bool, default=False, help='Whether to use GOF smoothing')
parser.add_argument('--cohort_year', type=int, default=1960, help='Cohort birth year')
parser.add_argument('--start_age', type=int, default=0, help='Start age')
parser.add_argument('--end_age', type=int, default=100, help='End age')
parser.add_argument('--cohort_sex', type=str, default='Male', help='Cohort sex')
parser.add_argument('--cohort_race', type=str, default='Black', help='Cohort race')
parser.add_argument('--num_patients', type=int, default=100_000, help='Number of patients')
parser.add_argument('--cancer_sites', nargs='+', default=['Lung', 'Colorectal', 'Pancreas', 'Prostate'], help='Cancer sites')
parser.add_argument('--cancer_sites_ed', nargs='+', default=['Lung', 'Colorectal', 'Pancreas', 'Prostate'], help='Cancer sites for early detection')

args, unknown = parser.parse_known_args()

def parse_args_into_config():
    if len(args.cancer_sites_ed[0])==0:
        CANCER_SITES_ED = []
    else:
        CANCER_SITES_ED = args.cancer_sites_ed
    args_dict = vars(args)
    args_dict["cancer_sites_ed"] = CANCER_SITES_ED
    return Config(**vars(args))

def run_model(cancer_sites: [""], cohort: int, sex: str, race: str, start_age = 0, end_age = 100, save=True):
    config = Config(mode="app",
                    save_results=False,
                    sojourn_time=False,
                    gof_smoothing=False,
                    cohort_year=cohort,
                    start_age=start_age,
                    end_age=end_age,
                    cohort_sex=sex,
                    cohort_race=race,
                    num_patients=100_000,
                    cancer_sites=cancer_sites,
                    cancer_sites_ed=[],
                    )
    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC = c.select_cohort(config)
    model = DiscreteEventSimulation(
        ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, len(cancer_sites), config)
    model.run(CANCER_PDF)
    inc_df = pd.DataFrame(model.cancerIncArr, columns=["Incidence per 100k"])
    inc_df = inc_df.reset_index().rename(columns={'index': 'Age'})

    # Output model incidence
    if save:
        app_dir = Path(__file__).parent / "app/data/incidence"
        inc_df.to_csv(f"{str(app_dir)}/{cohort}_{sex}_{race}_{cancer_sites[0]}.xlsx")
    return inc_df


def run_calibration(cohort, config: Config):
    # Initialize cohort-specific parameters
    ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC = c.select_cohort(config, cohort)
    model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, len(config.CANCER_SITES), config)

    # Run calibration 
    best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age, config)
    # Save as numpy file
    if config.SAVE_RESULTS:
        np.save(c.PATHS['calibration'] + f"{config.COHORT_SEX}_{config.COHORT_RACE}_{cohort}_{config.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
    try:
        plt.plot(np.arange(config.START_AGE, config.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
        plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.ylim(0, CANCER_INC.max() + 30)
        plt.title(f"Cancer Incidence by Age for Birthyear={cohort}, Sex={config.COHORT_SEX}, Race={config.COHORT_RACE}, Site={config.CANCER_SITES[0]}")
        plt.savefig(c.PATHS['plots_calibration'] + f"{config.COHORT_SEX}_{config.COHORT_RACE}_{cohort}_{config.CANCER_SITES[0]}.png", bbox_inches='tight')
        plt.clf()
    except Exception as e:
        print(f"An error occurred while plotting or saving the figure: {e}")

def main(config: Config = None):
    if not config:
        config = parse_args_into_config()

    if config.MODE == 'visualize':
        # Run simplest verion of the model: creates plot of model incidence vs SEER incidence
        # Runs for a single cancer site or multiple cancer sites
        print(f"RUNNING MODEL VISUALIZATION: COHORT = {config.COHORT_YEAR}, SEX = {config.COHORT_SEX}, RACE = {config.COHORT_RACE}")
        print(f"CANCERS = {config.CANCER_SITES}")
        # Initialize cohort-specific parameters
        if len(config.CANCER_SITES) == 1: # single cancer site
            ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC = c.select_cohort(config)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, len(config.CANCER_SITES), config)
            print(objective(model.run(CANCER_PDF).cancerIncArr, min_age, max_age, config.START_AGE, config.END_AGE, CANCER_INC))
        
            # Output model incidence, cancer count, alive count
            df = pd.DataFrame(model.cancerIncArr, columns = ['Incidence'])
            df['Cancer_Count'] = model.cancerCountArr
            df['Alive_Count'] = model.aliveCountArr
            df.to_excel(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{config.CANCER_SITES[0]}_SUMMARY.xlsx")
            
            if config.SOJOURN_TIME:
                with open(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{config.CANCER_SITES[0]}_LOG_sj.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)
            else:
                with open(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{config.CANCER_SITES[0]}_LOG_nh.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)

            # Limit the plot's y-axis to just above the highest SEER incidence
            plt.plot(np.arange(config.START_AGE, config.END_AGE), model.run(CANCER_PDF).cancerIncArr[:-1], label='Model', color='blue')
            plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
            plt.legend(loc='upper left')
            plt.ylim(0, CANCER_INC.max() + 30)
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={config.COHORT_YEAR}, Sex={config.COHORT_SEX}, Race={config.COHORT_RACE}, Site={config.CANCER_SITES[0]}")
            plt.savefig(c.PATHS['plots_calibration'] + f"{config.COHORT_SEX}_{config.COHORT_RACE}_{config.COHORT_YEAR}_{config.CANCER_SITES[0]}.png", bbox_inches='tight')
            plt.clf()
            plt.show()
        
        else: # multiple cancer sites
            ac_cdf, min_age, max_age, CANCER_PDF_lst, cancer_surv_arr_lst, cancer_surv_arr_lst_ed, sj_cancer_sites, CANCER_INC_lst = c.select_cohort(config)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr_lst, cancer_surv_arr_lst_ed, sj_cancer_sites, len(config.CANCER_SITES), config)
            plt.plot(np.arange(config.START_AGE, config.END_AGE), model.run(CANCER_PDF_lst).cancerIncArr[:-1], label='Model', color='blue')
            
            # Output model incidence, cancer count, alive count
            df = pd.DataFrame(model.cancerIncArr, columns = ['Incidence'])
            df['Cancer_Count'] = model.cancerCountArr
            df['Alive_Count'] = model.aliveCountArr
            df.to_excel(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{str(config.CANCER_SITES)}_SUMMARY.xlsx")
            
            if config.SOJOURN_TIME:
                cancer_sites_str = '_'.join(config.CANCER_SITES_ED)               
                with open(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{cancer_sites_str}_LOG_sj.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)
            else:
                cancer_sites_str = '_'.join(config.CANCER_SITES)
                with open(c.PATHS['output'] + f"{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{cancer_sites_str}_LOG_nh.pickle", 'wb') as handle:
                    pickle.dump(model.log, handle)

            plt.legend(loc='upper left')
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={config.COHORT_YEAR}, Sex={config.COHORT_SEX}, Race={config.COHORT_RACE}, Site={str(config.CANCER_SITES)}")
            plt.show()

    elif config.MODE == 'app':
        # app_dir = Path(__file__).parent / "app/data/incidence"
        # filename = f"{str(app_dir)}/{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{str(c.CANCER_SITES)}.csv"
        
        # Initialize cohort-specific parameters
        ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC = c.select_cohort(config)
        model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, len(config.CANCER_SITES), config)
        model.run(CANCER_PDF).cancerIncArr
    
        # Output model incidence, cancer count, alive count
        df = pd.DataFrame(model.cancerIncArr, columns = ['Incidence per 100k'])
        df['Cancer_Count'] = model.cancerCountArr
        df['Alive_Count'] = model.aliveCountArr
        df = df.reset_index().rename(columns={'index': 'Age'})

        # Get CSV string and print to stdout to be parsed by frontend
        print(df.to_csv())

    elif config.MODE == 'cancer_dist': # Saves a plot of the calibrated cancer cdf and pdf
        # Initialize cohort-specific parameters
        ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, CANCER_INC = c.select_cohort(config)
        # Plot pdf
        fig = plt.figure(figsize = (10,5))
        plt.bar(range(config.START_AGE, config.END_AGE+1), CANCER_PDF) # doesn't start at age 0, starts at age 1
        plt.xlabel("Age")
        plt.ylabel("PDF")
        plt.savefig(c.PATHS['plots'] + f"cancer_pdf_{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{config.CANCER_SITES[0]}.png")
        plt.clf()

        fig = plt.figure(figsize = (10,5))
        plt.bar(range(config.START_AGE, config.END_AGE+1), np.cumsum(CANCER_PDF)) # doesn't start at age 0, starts at age 1
        plt.xlabel("Age")
        plt.ylabel("CDF")
        plt.savefig(c.PATHS['plots'] + f"cancer_cdf_{config.COHORT_YEAR}_{config.COHORT_SEX}_{config.COHORT_RACE}_{config.CANCER_SITES[0]}.png")
        plt.clf()

    elif config.MODE == 'calibrate':
        if config.MULTI_COHORT_CALIBRATION:
            print(f"RUNNING MULTI-COHORT CALIBRATION: FIRST COHORT = {config.FIRST_COHORT}, LAST COHORT = {config.LAST_COHORT}, SEX = {config.COHORT_SEX}, RACE = {config.COHORT_RACE}")
            print(f"CANCERS = {config.CANCER_SITES}")

            run_calibration_with_config = partial(run_calibration, config=config)
            with mp.Pool(processes=config.NUM_PROCESSES) as pool:
                results = list(tqdm(pool.imap(run_calibration_with_config, range(config.FIRST_COHORT, config.LAST_COHORT + 1)), total=config.LAST_COHORT - config.FIRST_COHORT + 1))

        else:
            print(f"RUNNING CALIBRATION: COHORT = {config.COHORT_YEAR}, SEX = {config.COHORT_SEX}, RACE = {config.COHORT_RACE}")
            print(f"CANCERS = {config.CANCER_SITES}")
            # Initialize cohort-specific parameters
            ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC = c.select_cohort(config)
            model = DiscreteEventSimulation(ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, len(config.CANCER_SITES), config)
            # Run calibration for simplest verion of the model
            best = simulated_annealing(model, CANCER_PDF, CANCER_INC, min_age, max_age, config)
            # Save as numpy file, time_stamped
            if config.SAVE_RESULTS:
                np.save(config.PATHS['calibration'] + f"{config.COHORT_SEX}_{config.COHORT_RACE}_{config.COHORT_YEAR}_{config.CANCER_SITES[0]}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)
            # Limit the plot's y-axis to just above the highest SEER incidence
            plt.plot(np.arange(config.START_AGE, config.END_AGE), model.run(best).cancerIncArr[:-1], label='Model', color='blue')
            plt.plot(np.arange(min_age, max_age+1), CANCER_INC, label='SEER', color='darkred', alpha=0.5)
            plt.legend(loc='upper left')
            plt.ylim(0, CANCER_INC.max() + 30)
            plt.xlabel('Age')
            plt.ylabel('Incidence (per 100k)')
            plt.title(f"Cancer Incidence by Age for Birthyear={config.COHORT_YEAR}, Sex={config.COHORT_SEX}, Race={config.COHORT_RACE}, Site={config.CANCER_SITES[0]}")
            plt.savefig(c.PATHS['plots_calibration'] + f"{config.COHORT_SEX}_{config.COHORT_RACE}_{config.COHORT_YEAR}_{config.CANCER_SITES[0]}.png", bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    start = timer()
    main()
    end = timer()
    print(f'total time: {timedelta(seconds=end-start)}')



