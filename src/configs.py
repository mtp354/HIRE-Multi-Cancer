# Configurations of the multi-cancer model
import numpy as np
import pandas as pd
import glob
import os
from pathlib import Path
import random

# Aesthetic Preferences
np.set_printoptions(precision=5, suppress=True)

class Config:
    def __init__(self,
                 mode="visualize",
                 save_results=True,
                 sojourn_time=False,
                 gof_smoothing=False,
                 cohort_year=1960,
                 start_age=0,
                 end_age=100,
                 cohort_sex="Male",
                 cohort_race="Black",
                 num_patients=100_000,
                 cancer_sites=['Lung', 'Colorectal', 'Pancreas', 'Prostate'],
                 cancer_sites_ed=['Lung', 'Colorectal', 'Pancreas', 'Prostate'],
                 ):
        """Defines model run/calibration configs"""

        self.MODE = mode
        # Options:
        # - calibrate: run simulated annealing for cancer incidence (one site)
        # - visualize: plot incidence and mortality, output cancer incidence, cancer count, alive count
        # - cancer_dist: plot cancer pdf and cdf
        self.SAVE_RESULTS = save_results # whether to save results to file
        self.SOJOURN_TIME = sojourn_time
        self.GOF_SMOOTHING = gof_smoothing # whether to add smoothing to model incidence during calibration
        # Define cohort characteristics
        self.COHORT_YEAR = cohort_year  # birth year of the cohort
        self.START_AGE = start_age
        self.END_AGE = end_age
        self.COHORT_SEX = cohort_sex  # Female/Male
        self.COHORT_RACE = cohort_race  # Black/White
        self.NUM_PATIENTS = num_patients
        self.CANCER_SITES = cancer_sites
        self.CANCER_SITES_ED = cancer_sites_ed # cancer types that have screening methods for the early detection 

        ## Multiprocessing
        self.NUM_PROCESSES = 12

        # Define simulated annealing parameter
        self.NUM_ITERATIONS = 2_000
        self.START_TEMP = 10
        self.STEP_SIZE = 0.01 #0.001
        self.VERBOSE = True
        self.MASK_SIZE = 0.5 # value between 0 and 1, the fraction of values to modify each step
        self.LOAD_LATEST = True # If true, load the latest cancer_pdf from file as starting point
        # LOAD_LATEST is used to get the most recently calibrated numpy file to run the model
        # First checks if there is a previous file for same sex/race/cancer site, then same sex/cancer site,
        # then same race/cancer site, then same cancer site

        # Note: You can either do multi-calibration by increasing or decreasing cohort years
        # Range is based on FIRST_COHORT and LAST_COHORT
        # To do calibration in ascending cohort years, you MUST have a starting numpy file for the FIRST_COHORT or else
        # LOAD_LATEST cannot work with MULTI_COHORT_CALIBRATION correctly
        # So generally you should set both LOAD_LATEST and MULTI_COHORT_CALIBRATION to True
        # You MUST do multi-calibration in ascending order FIRST before doing descending order
        # You CANNOT start multi-cohort calibration in descending order first
        # When you do reverse calibration, remember that the LAST_COHORT looks at the next +1 birth year cohort year
        self.MULTI_COHORT_CALIBRATION = False
        self.REVERSE_MULTI_COHORT_CALIBRATION = False # determines whether you want to reverse the cohort year range in calibration
        if self.MULTI_COHORT_CALIBRATION:
            self.FIRST_COHORT = 1935
            self.LAST_COHORT = 1960

        ## Random numbers
        self.random_numbers_array = np.random.rand(self.NUM_PATIENTS, 100)
        self.rand4step = np.random.randint(0, 100000, size=self.NUM_ITERATIONS)

        self.check_params_valid()

    def check_params_valid(self):
        """Check if param set is valid

        Raises:
            ValueError: Cannot have REVERSE_MULTI_COHORT_CALIBRATION set to True while MULTI_COHORT_CALIBRATION is set to False
            ValueError: Cannot set MULTI_COHORT_CALIBRATION to True for a MODE other than calibrate
            Exception: Cancer site and cohort sex combination is not valid
            Exception: Cannot calibrate multiple cancer sites at the same time
            Exception: Can only run cancer_dist for one cancer site
        """
        if self.REVERSE_MULTI_COHORT_CALIBRATION == True and self.MULTI_COHORT_CALIBRATION == False:
            raise ValueError("ERROR: You cannot have REVERSE_MULTI_COHORT_CALIBRATION set to True while MULTI_COHORT_CALIBRATION is set to False")
        if self.MULTI_COHORT_CALIBRATION and self.MODE != "calibrate":
            raise ValueError("ERROR: You cannot set MULTI_COHORT_CALIBRATION to True for a MODE other than calibrate")
        # Raise exceptions for male/ovarian, male/uterine, male/cervical, female/prostrate
        if self.COHORT_SEX == 'Male' and ('Ovarian' in self.CANCER_SITES or 'Uterine' in self.CANCER_SITES or 'Cervical' in self.CANCER_SITES):
            raise Exception("Cancer site and cohort sex combination is not valid in configs.py")
        elif self.COHORT_SEX == 'Female' and 'Prostate' in self.CANCER_SITES:
            raise Exception("Cancer site and cohort sex combination is not valid in configs.py")

        # Raise exceptions for other modes of the model
        if len(self.CANCER_SITES) > 1 and self.MODE == 'calibrate':
            raise Exception("You cannot calibrate multiple cancer sites at the same time")
        if len(self.CANCER_SITES) > 1 and self.MODE == 'cancer_dist':
            raise Exception("You can only run cancer_dist for one cancer site")
    
    def __repr__(self) -> str:
        if self.MODE == "calibration":
            return f"CONFIG = {"REVERSE " if self.REVERSE_MULTI_COHORT_CALIBRATION else ""}{"MULTI-COHORT " if self.MULTI_COHORT_CALIBRATION else ""}{self.MODE}. CANCER SITE: {self.CANCER_SITES[0]}, COHORT: {f"{self.FIRST_COHORT}-{self.LAST_COHORT}" if self.MULTI_COHORT_CALIBRATION else self.COHORT_YEAR}, SEX: {self.COHORT_SEX}, RACE: {self.COHORT_RACE}, AGE: {f"{self.START_AGE}-{self.END_AGE}"}, NUM PATIENTS: {self.NUM_PATIENTS}. CALIBRATION -- NUM ITERATIONS {self.NUM_ITERATIONS}, LOAD LATEST {self.LOAD_LATEST}, NUM PROCESSES: {self.NUM_PROCESSES}, START TEMP: {self.START_TEMP}, STEP SIZE: {self.STEP_SIZE}, MASK SIZE: {self.MASK_SIZE}, SMOOTHING: {self.GOF_SMOOTHING}."
        return f"Config = {self.MODE}. CANCER SITE(s): {self.CANCER_SITES}, COHORT: {self.COHORT_YEAR}, SEX: {self.COHORT_SEX}, RACE: {self.COHORT_RACE}, AGE: {f"{self.START_AGE}-{self.END_AGE}"}, NUM PATIENTS: {self.NUM_PATIENTS}. EARLY DETECTION -- CANCER SITES ED: {self.CANCER_SITES_ED}, SOJOURN TIME {self.SOJOURN_TIME}."


# Define input and output paths
parent_dir = Path(__file__).parent.parent
PATHS = {
    "incidence": str(parent_dir / "data/cancer_incidence") + "/",
    "mortality": str(parent_dir / "data/mortality") + "/",
    "survival": str(parent_dir / "data/cancer_survival") + "/",
    "calibration": str(parent_dir / "outputs/calibration") + "/",
    "plots_calibration": str(parent_dir / "outputs/calibration/plots") + "/",
    "sojourn_time": str(parent_dir / "data/Sojourn Times") + "/",
    "plots": str(parent_dir / "outputs/plots") + "/",
    "output": str(parent_dir / "outputs") + "/",
}


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Selecting Cohort
def select_cohort(config: Config, birthyear=None):
    print(config)
    birthyear = config.COHORT_YEAR if birthyear == None else birthyear
    # Load input data
    CANCER_INC = pd.read_csv(f'{PATHS["incidence"]}Incidence.csv')
    CANCER_INC = CANCER_INC[CANCER_INC['Site'].isin(config.CANCER_SITES)]  # keeping the cancers of interest

    # Load in mortality data
    MORT = pd.read_csv(f'{PATHS["mortality"]}Mortality.csv')
    MORT = MORT[~MORT['Site'].isin(config.CANCER_SITES)]  # Removing the cancers of interest
    MORT = MORT.groupby(['Cohort','Age','Sex','Race']).agg({'Rate':'sum'}).reset_index()  # Summing over the remaining sites

    # Load in Survival data # TODO: no male breast survival data
    SURV = pd.read_csv(f'{PATHS["survival"]}Survival.csv')  # This is the 10 year survival by cause
    SURV = SURV[SURV['Site'].isin(config.CANCER_SITES)]  # keeping the cancers of interest
    
    # Load in Survival_Localized
    SURV_ed = pd.read_csv(f'{PATHS["survival"]}Survival_Localized.csv')  # This is the 10 year survival by cause
    SURV_ed = SURV_ed[SURV_ed['Site'].isin(config.CANCER_SITES_ED)]  # keeping the cancers of interest

    sex = config.COHORT_SEX
    race = config.COHORT_RACE[3:]
    CANCER_INC.query('Sex == @sex & Race == @race & Cohort == @birthyear', inplace=True)
    
    # Impute SEER data using earlier cohorts upto age 83
    if CANCER_INC['Age'].max() < 83:
        i=1
        while CANCER_INC['Age'].max()<83:
            cohort_year = birthyear-i
            cancer_inc = pd.read_csv(f'{PATHS["incidence"]}Incidence.csv')
            cancer_inc = cancer_inc[cancer_inc['Site'].isin(config.CANCER_SITES)]  # keeping the cancers of interest
            cancer_inc.query('Sex == @sex & Race == @race & Cohort == @cohort_year', inplace=True)
            CANCER_INC = pd.concat([CANCER_INC, cancer_inc.iloc[-1,:].to_frame().T])
            i = i+1

    # CANCER_INC = CANCER_INC.iloc[:-4,:] # when you need to adjust maximum age
    # Add linear line from anchoring point to the age at first incidence data point
    if list(CANCER_INC['Age'])[0] > 18:
        fillup_age = list(range(18, list(CANCER_INC['Age'])[0]))
        slope = list(CANCER_INC['Rate'])[0]/(list(CANCER_INC['Age'])[0]-18)
        intercept = -18*slope
        fillup_rate = slope*np.array(fillup_age)+intercept
        fillup_df = pd.DataFrame({'Age': fillup_age, 'Rate': fillup_rate})
        CANCER_INC = pd.concat([fillup_df, CANCER_INC])

        min_age = 18
        # max_age = min(2018 - birthyear, 83)
    else:
        # For plotting and objective, we only compare years we have data
        min_age = max(1975 - birthyear, 0)
        # max_age = min(2018 - birthyear, 83)
    # min_age = 18
    max_age = 83 # max_age needs to be 83 as we are imputing SEER data to age 83
    
    MORT.query('Sex == @sex & Race == @race & Cohort == @birthyear', inplace=True)
    SURV.query('Sex == @sex & Race == @race', inplace=True)
    SURV_ed.query('Sex == @sex & Race == @race', inplace=True)

    # Create the cdf all-cause mortality
    ac_cdf = np.cumsum(MORT['Rate'].to_numpy())/100000
    # Creating the conditional CDF for all-cause mortality by age (doing this here to save runtime)
    ac_cdf = np.tile(ac_cdf, (config.END_AGE - config.START_AGE + 1, 1))  # (current age, future age)
    
    for i in range(config.END_AGE - config.START_AGE + 1):
        ac_cdf[i, :] -= ac_cdf[i, i]  # Subtract the death at current age
    ac_cdf[:, -1] = 1.0  # Adding 1.0 to the end to ensure death at 100
    ac_cdf = np.clip(ac_cdf, 0.0, 1.0)
    
    # Load all cancer incidence target data
    if len(config.CANCER_SITES) == 1: # only 1 cancer site
        CANCER_INC = CANCER_INC['Rate'].to_numpy()
    else: # multiple cancer sites
        # Need to separate each cancer incidence
        # Add numpy array for each cancer site into a lst
        CANCER_INC_lst = []
        for i in range(len(config.CANCER_SITES)):
            temp = CANCER_INC[CANCER_INC['Site']==config.CANCER_SITES[i]]
            tempArr = temp['Rate'].to_numpy()
            CANCER_INC_lst.append(tempArr)

    # Loading in cancer pdf, this is the thing that will be optimized over
    CANCER_PDF_DEFAULT = 0.002 * np.ones(config.END_AGE - config.START_AGE + 1)  # starting from 0 incidence and using bias optimization
    CANCER_PDF_DEFAULT[:min(config.END_AGE - config.START_AGE + 1, max(0, 35 - config.START_AGE))] = 0.0 # set ages before 35 to 0.0
    
    CANCER_PDF = CANCER_PDF_DEFAULT
    CANCER_PDF_lst = []

    if config.LOAD_LATEST:
        for i in range(len(config.CANCER_SITES)):
            # Check if there is a previous numpy file matching the same sex and race and cancer site
            list_of_files = glob.glob(f'{PATHS["calibration"]}*{config.COHORT_SEX}_{config.COHORT_RACE}_*{config.CANCER_SITES[i]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same sex and cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{config.COHORT_SEX}_*{config.CANCER_SITES[i]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same race and cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{config.COHORT_RACE}_*{config.CANCER_SITES[i]}_*.npy')
            if len(list_of_files) == 0: # Check if there is a previous numpy file matching the same cancer site
                list_of_files = glob.glob(f'{PATHS["calibration"]}*{config.CANCER_SITES[i]}_*.npy')
            # if len(list_of_files) == 0:
            #     raise ValueError("No suitable LOAD_LATEST file, set LOAD_LATEST to FALSE")

            if len(list_of_files):
                # Look at all the unique cohort years in the file names
                all_cohort_years = []
                for file in list_of_files:
                    year = file.split('_')[2] # grabs the cohort year
                    if int(year) not in all_cohort_years:
                        all_cohort_years.append(int(year))
                if config.MULTI_COHORT_CALIBRATION == False or config.REVERSE_MULTI_COHORT_CALIBRATION == False: # ascending birth year calibration
                    # Sort ascending years
                    all_cohort_years.sort()
                    # Get the max calibrated cohort year that is just below or equal to the COHORT_YEAR
                    max_year = None
                    for year in all_cohort_years:
                        if year <= birthyear:
                            max_year = year
                    final_list = []
                    for file in list_of_files:
                        if f'_{max_year}_' in file:
                            final_list.append(file)
                elif config.MULTI_COHORT_CALIBRATION and config.REVERSE_MULTI_COHORT_CALIBRATION: # descending birth year calibration
                    # Get the min calibrated cohort year that is just above the COHORT_YEAR
                    min_year = birthyear + 1
                    final_list = []
                    for file in list_of_files:
                        if f'_{min_year}_' in file:
                            final_list.append(file)
            # else:
            #     raise ValueError("ERROR: LOAD_LATEST fails in configs.py")
            ## Read the latest file
            latest_file = max(final_list, key=os.path.getctime) if len(final_list) else None
            CANCER_PDF = np.load(latest_file) if latest_file else CANCER_PDF_DEFAULT
            CANCER_PDF = CANCER_PDF[config.START_AGE:config.END_AGE+1]
            CANCER_PDF_lst.append(CANCER_PDF)

    # Loading in cancer survival data
    if len(config.CANCER_SITES) == 1: # 1 cancer site
        SURV = SURV[['Cancer_Death','Other_Death']].to_numpy()[config.START_AGE:config.END_AGE+1]  # 10 year survival
        SURV = 1 - SURV**(1/10)  # Converting to annual probability of death (assuming constant rate)

        SURV_ed = SURV_ed[['Cancer_Death','Other_Death']].to_numpy()[config.START_AGE:config.END_AGE+1]  # 10 year survival
        SURV_ed = 1 - SURV_ed**(1/10)  # Converting to annual probability of death (assuming constant rate)
        
        # Converting into probability of death at each follow up year
        cancer_surv_arr = np.zeros((config.END_AGE - config.START_AGE + 1, 10, 2))
        cancer_surv_arr_ed = np.zeros((config.END_AGE - config.START_AGE + 1, 10, 2))

        for i in range(10):
            cancer_surv_arr[:,i,0] = 1-(1-SURV[:,0])**(i+1)  # Cancer death
            cancer_surv_arr[:,i,1] = 1-(1-SURV[:,1])**(i+1)  # Other death
            
            cancer_surv_arr_ed[:,i,0] = 1-(1-SURV[:,0])**(i+1)  # Cancer death
            cancer_surv_arr_ed[:,i,1] = 1-(1-SURV[:,1])**(i+1)  # Other death
            # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year
    else: # multiple cancer sites
        cancer_surv_arr_lst = []
        for i in range(len(config.CANCER_SITES)):
            temp = SURV[SURV['Site']==config.CANCER_SITES[i]][config.START_AGE:config.END_AGE+1]
            temp = temp[['Cancer_Death','Other_Death']].to_numpy()  # 10 year survival
            temp = 1 - temp**(1/10)  # Converting to annual probability of death (assuming constant rate)

            # Converting into probability of death at each follow up year
            cancer_surv_arr = np.zeros((config.END_AGE - config.START_AGE + 1, 10, 2))

            for i in range(10):
                cancer_surv_arr[:,i,0] = 1-(1-temp[:,0])**(i+1)  # Cancer death
                cancer_surv_arr[:,i,1] = 1-(1-temp[:,1])**(i+1)  # Other death
                # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year

            cancer_surv_arr_lst.append(cancer_surv_arr)
            
        cancer_surv_arr_lst_ed = []
        for i in range(len(config.CANCER_SITES_ED)):
            temp = SURV_ed[SURV_ed['Site']==config.CANCER_SITES_ED[i]][config.START_AGE:config.END_AGE+1]
            temp = temp[['Cancer_Death','Other_Death']].to_numpy()  # 10 year survival
            temp = 1 - temp**(1/10)  # Converting to annual probability of death (assuming constant rate)

            # Converting into probability of death at each follow up year
            cancer_surv_arr_ed = np.zeros((config.END_AGE - config.START_AGE + 1, 10, 2))

            for i in range(10):
                cancer_surv_arr_ed[:,i,0] = 1-(1-temp[:,0])**(i+1)  # Cancer death
                cancer_surv_arr_ed[:,i,1] = 1-(1-temp[:,1])**(i+1)  # Other death
                # This is now an an array of shape (100, 10, 2), that represents the cdf of cancer death and other death at each follow up year

            cancer_surv_arr_lst_ed.append(cancer_surv_arr_ed)

    # Load in sojorn times
    sojourn = pd.read_csv(PATHS['sojourn_time'] + 'Sojourn Estimates.csv')
    sj_cancer_sites = {}
    for i in range(len(config.CANCER_SITES)):
        s = sojourn[sojourn['Site'].isin([config.CANCER_SITES[i]])]
        sj_cancer_sites[i] = np.random.triangular(s['Lower'], s['Sojourn Time'], s['Upper'], config.NUM_PATIENTS).astype(int)

    if len(config.CANCER_SITES) == 1:
        return ac_cdf, min_age, max_age, CANCER_PDF, cancer_surv_arr, cancer_surv_arr_ed, sj_cancer_sites, CANCER_INC
    else:
        return ac_cdf, min_age, max_age, CANCER_PDF_lst, cancer_surv_arr_lst, cancer_surv_arr_lst_ed, sj_cancer_sites, CANCER_INC_lst
    # If we are running the model for multiple cancers, each cancer is a separate element in a list