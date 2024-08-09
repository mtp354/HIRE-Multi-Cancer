# Defines all classes and functions used in the simulation
import numpy as np
import configs as c
from sklearn.metrics import mean_squared_error
from csaps import csaps
from tqdm import tqdm
import random
import pickle
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
        
class Patient:
    def __init__(self, pid, num_cancers, starting_age=c.START_AGE, rn_pid = None, detected_early = 0):
        """
        Initializes the object with the given `pid`, `age`, `cancer_pdf`.

        :param pid: The patient ID.
        :param num_cancers: The number of cancer sites
        :param age: The age of the patient.
        :param cancer_pdf: The PDF file containing information about cancer. This can be a list if there are multiple cancers.

        :return: None
        """
        self.pid = pid
        self.age = starting_age
        self.current_state = 'Healthy'
        self.history = [{self.current_state:self.age}]  # A dictionary to store the state and the age at entry to the state
        self.num_cancers = num_cancers
        self.cancer_type = None # only used for multiple cancers
        self.detected_early = detected_early
        self.random_numbers = rn_pid
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Patient object for debugging purposes.
        This function does not take any parameters.
        It returns a string.
        """
        return f"Patient(pid={self.pid}, age={self.age}, history={self.history})"

    def __str__(self) -> str:
        """
        Return a simplified string representation of the Patient object's attributes.
        """
        return f"Patient:{self.pid}, history:{self.history})"
    
    def run(self, cancer_pdf, ac_cdf, cancer_surv_arr, cancer_surv_arr_ed):
        self.reset()  
        
        # store random numbers
        # make sure new random numbers are used for every searchsorted
        r=0    
        while 'Death' not in self.current_state and self.age<c.END_AGE:
            if self.current_state == 'Healthy':
                time_to_od = np.searchsorted(ac_cdf[self.age - c.START_AGE,:], self.random_numbers[r]) - self.age
                r = r+1

                if self.num_cancers == 1:
                    time_to_cancer = np.searchsorted(np.cumsum(cancer_pdf), self.random_numbers[r])
                    r = r+1
                    if time_to_cancer <= time_to_od:  # If cancer happens before death
                        self.current_state = 'Cancer'
                        self.cancer_type = c.CANCER_SITES[0]
                        self.age += time_to_cancer
                    else:
                        self.current_state = 'Other Death'
                        self.age += time_to_od

                else:
                    # Create a numpy array to hold all the time_to_cancer for each cancer site
                    time_to_cancer_arr = np.zeros((self.num_cancers))
                    for i in range(self.num_cancers): # in this case cancer_pdf is a list
                        temp_time = np.searchsorted(np.cumsum(cancer_pdf[i]), self.random_numbers[r])
                        r = r+1
                        time_to_cancer_arr[i] += temp_time
                    
                    # Find the earliest age out of the all the cancer times
                    time_to_cancer = int(time_to_cancer_arr.min())
                    i_time_to_cancer = time_to_cancer_arr.argmin() # index of min time

                    if time_to_cancer <= time_to_od:  # If cancer happens before death
                        self.current_state = 'Cancer'
                        self.cancer_type = c.CANCER_SITES[i_time_to_cancer]
                        self.age += time_to_cancer
                    else:
                        self.current_state = 'Other Death'
                        self.age += time_to_od
                        

                if self.detected_early==1 and self.current_state == 'Cancer' and self.cancer_type in c.CANCER_SITES_ED:
                    if self.num_cancers == 1:
                        self.age -= c.sj_cancer_sites[0][self.pid]
                    else:
                        self.age -= c.sj_cancer_sites[i_time_to_cancer][self.pid]
                    self.age = max(0, self.age)
                    self.history.append({self.current_state:(self.age, self.cancer_type)})
                    self.current_state = 'Cancer'
                    # self.history.append({self.current_state:self.age})
                else:
                    if self.current_state == 'Cancer' and self.num_cancers == 1:
                        self.history.append({self.current_state:(self.age, self.cancer_type, c.sj_cancer_sites[0][self.pid])})
                    elif self.current_state == 'Cancer' and self.num_cancers>1:
                        self.history.append({self.current_state:(self.age, self.cancer_type, c.sj_cancer_sites[i_time_to_cancer][self.pid])})
                    else:
                        self.history.append({self.current_state: self.age})
            
            if self.current_state == 'Cancer':
                time_at_risk = min(10, c.END_AGE-self.age-1)

                # If running multiple cancers, make sure to select the correct cancer_surv_array

                if self.num_cancers == 1:
                    if self.detected_early==1 and self.cancer_type in c.CANCER_SITES_ED:
                        time_to_cd = np.searchsorted(cancer_surv_arr_ed[self.age - c.START_AGE, :1+time_at_risk, 0], self.random_numbers[r])
                        r = r+1
                        time_to_od = np.searchsorted(cancer_surv_arr_ed[self.age - c.START_AGE, :1+time_at_risk, 1], self.random_numbers[r])
                        r = r+1
                    else:
                        time_to_cd = np.searchsorted(cancer_surv_arr[self.age - c.START_AGE, :1+time_at_risk, 0], self.random_numbers[r])
                        r = r+1
                        time_to_od = np.searchsorted(cancer_surv_arr[self.age - c.START_AGE, :1+time_at_risk, 1], self.random_numbers[r])
                        r = r+1
                else:
                    if self.detected_early==1 and self.cancer_type in c.CANCER_SITES_ED:
                        time_to_cd = np.searchsorted(cancer_surv_arr_ed[c.CANCER_SITES.index(self.cancer_type)][self.age - c.START_AGE, :1+time_at_risk, 0], self.random_numbers[r])
                        r = r+1
                        time_to_od = np.searchsorted(cancer_surv_arr_ed[c.CANCER_SITES.index(self.cancer_type)][self.age - c.START_AGE, :1+time_at_risk, 1], self.random_numbers[r])                        
                        r = r+1                        
                    else:
                        time_to_cd = np.searchsorted(cancer_surv_arr[c.CANCER_SITES.index(self.cancer_type)][self.age - c.START_AGE, :1+time_at_risk, 0], self.random_numbers[r])
                        r = r+1
                        time_to_od = np.searchsorted(cancer_surv_arr[c.CANCER_SITES.index(self.cancer_type)][self.age - c.START_AGE, :1+time_at_risk, 1], self.random_numbers[r])                        
                        r = r+1
                if time_to_od < time_to_cd:  # # If other death happens before cancer
                    self.current_state = 'Other Death'
                    self.age += time_to_od
                elif time_to_cd < time_at_risk:  # If cancer death happens before other death
                    self.current_state = 'Cancer Death'
                    self.age += time_to_cd
                else:
                    self.current_state = 'Healthy'
                    self.age += time_at_risk
                self.history.append({self.current_state: self.age})     
        return self.history

    def reset(self):
        """
        Reset the state of the object by setting history to contain the current age and 0, and 
        setting karma to a random value between 0 and 1.
        """
        self.age = c.START_AGE
        self.current_state = 'Healthy'
        self.history = [{self.current_state:self.age}]
        self.cancer_type = None # only used for multiple cancers

class DiscreteEventSimulation:
    def __init__(self, ac_cdf, cancer_surv_arr, cancer_surv_arr_ed, num_cancers, num_patients=c.NUM_PATIENTS, starting_age=c.START_AGE, sojourn_time = c.SOJOURN_TIME):
        """
        Initializes the object with the given `cancer_cdf`.
        """
        self.num_patients = num_patients
        random_numbers_array = c.random_numbers_array
        self.patients = []
        if sojourn_time:
            cancer_sites_str = '_'.join(c.CANCER_SITES)
            with open(c.PATHS['output'] + f"{c.COHORT_YEAR}_{c.COHORT_SEX}_{c.COHORT_RACE}_{cancer_sites_str}_LOG_nh.pickle", 'rb') as handle:
                nh = pickle.load(handle)
            for pid in range(self.num_patients):
                status = [list(s.keys())[0] for s in nh[pid]]
                    
                if 'Cancer' in status:
                    self.patients.append(Patient(pid, num_cancers, starting_age, rn_pid = random_numbers_array[pid, :], detected_early = 1))
                else:
                    self.patients.append(Patient(pid, num_cancers, starting_age, rn_pid = random_numbers_array[pid, :], detected_early = 0))
        else:
            for pid in range(self.num_patients):
                self.patients.append(Patient(pid, num_cancers, starting_age, rn_pid = random_numbers_array[pid, :], detected_early = 0))
        self.log = []  # A log of all patient dictionaries
        self.num_cancers = num_cancers # number of cancer sites we are running

        self.cancerIncArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize incidence array
        self.acMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize mortality array
        self.cancerMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize cancer mortality array
        self.ac_cdf = ac_cdf
        self.cancer_surv_arr = cancer_surv_arr
        self.cancer_surv_arr_ed = cancer_surv_arr_ed

        # Counts
        self.cancerCountArr = np.zeros((c.END_AGE - c.START_AGE + 1)) # all cancers
        self.aliveCountArr = np.zeros((c.END_AGE - c.START_AGE + 1))
        
    def generate_random_numbers(self, num):
        ran_nums = []
        for i in range(num):
            ran_nums.append(np.random.rand())
        return ran_nums
    
    def run(self, cancer_pdf):
        """
        Runs the discrete event simulation for the given number of patients.
        """
        self.reset()
        
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)

        for patient in self.patients: # runs for each patient
            patient_history = patient.run(cancer_pdf, self.ac_cdf, self.cancer_surv_arr, self.cancer_surv_arr_ed)  # running patient
            self.log.append(patient_history)  # recording to log
            # print(patient_history)
            try:
                # if len([1 for h in patient_history if list(h.keys())[0]=='Cancer'])>1: # if value of key "Cancer" is tuple
                age_at_cancer = [h['Cancer'][0] for h in patient_history if list(h.keys())[0]=='Cancer']
                for a in age_at_cancer:
                    self.cancerIncArr[a - c.START_AGE] += 1  # Increment the incidence count for the corresponding age
                # else:
                    # self.cancerIncArr[patient_history['Cancer'] - c.START_AGE] += 1  # Increment the incidence count for the corresponding age
                   
            except KeyError: 
                pass
            try:
                self.acMortArr[patient_history[-1]['Other Death'] - c.START_AGE] += 1  # Increment the mortality count for the corresponding age
            except KeyError: 
                pass
            try:
                self.cancerMortArr[patient_history[-1]['Cancer Death'] - c.START_AGE] += 1  # Increment the cancer mortality count for the corresponding age
            except KeyError: 
                pass
        num_alive = self.num_patients - self.acMortArr.cumsum() - self.cancerMortArr.cumsum()  # Adjusting denominator based on number alive
        self.aliveCountArr = num_alive
        self.cancerCountArr = self.cancerIncArr.copy()
        self.cancerIncArr =100000*np.divide(self.cancerIncArr, num_alive+1)  # adding 1 to avoid NaNs
        return self
    
    def reset(self):
        """
        Reset the object.
        """
        self.log = []
        self.cancerIncArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize incidence array
        self.acMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize mortality array
        self.cancerMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize cancer mortality array


# Defining Simulated Annealing functions
def objective(obs, min_age, max_age, exp):
    """
    A function that calculates the mean squared error between observed and expected values.

    Parameters:
    obs (array-like): The observed values.
    min_age (int): min age we have seer data for
    max_age (int): max age we have seer data for
    exp (array-like, optional): The expected values, default is c.CANCER_INC.

    Returns:
    float: The mean squared error between the observed and expected values.
    """
    # Calculate gof for age range intersection.
    # Ex: (start age, end age) = (20, 100). (min_age, max_age) = (18, 83). Want to calculate fit for (20, 83).
    start = max(min_age, c.START_AGE)
    end = min(max_age, c.END_AGE)
    return mean_squared_error(obs[start - c.START_AGE: end + 1 - c.START_AGE],
                              exp[start - min_age : end + 1 - min_age])

def step(candidate, step_size=c.STEP_SIZE, mask_size=c.MASK_SIZE, n_th_iteration = 0):
    """
    Generate a new candidate by adding random noise to the input candidate array, and then clipping the values to be within the range of 0.0 and 1.0.
    Parameters:
    - candidate: The input array of values.
    - step_size: The size of the random noise to be added to the candidate array.
    Returns:
    - The new candidate array with values clipped between 0.0 and 1.0.
    """
    x=c.rand4step[n_th_iteration]
    randVal_gen = np.random.RandomState(x)
    mask =  randVal_gen.random_sample(candidate.shape) > mask_size # fraction of values to modify
    perturbations = (randVal_gen.random_sample(mask.sum()) * 2 - 1) * step_size
    candidate[mask] += perturbations
    total_ages = c.END_AGE - c.START_AGE + 1
    candidate[max(0, 0 - c.START_AGE):min(total_ages, max(0, 18 - c.START_AGE + 1))] = 0.0  # anchoring
    candidate[max(0, 83 - c.START_AGE):min(total_ages, max(0, 100 - c.START_AGE + 1))] = 0.0
    ages_linspace = np.linspace(c.START_AGE, c.END_AGE, total_ages)
    candidate = csaps(ages_linspace, candidate, smooth=0.0001)(ages_linspace).clip(0.0, 1.0)   # smoothing 0.0001
    return candidate

def smooth_incidence(model_incid, window_length, polynomial):
    # Smooth model incidence
    smoothed_incid = savgol_filter(model_incid, window_length, polynomial)
    return smoothed_incid

def simulated_annealing(des, cancer_pdf, cancer_inc, min_age, max_age, n_iterations=c.NUM_ITERATIONS, 
                        start_temp=c.START_TEMP, step_size=c.STEP_SIZE, mask_size=c.MASK_SIZE, verbose=c.VERBOSE):
    """
    Simulated annealing algorithm to optimize a given cancer probability density function.

    Args:
        dse (obj): The instance of the dse class.
        cancer_pdf (numpy array): The initial cancer probability density function.
        n_iterations (int): The number of iterations for the algorithm.
        step_size (float): The size of the step for each iteration.
        verbose (bool, optional): Whether to display progress. Defaults to False.

    Returns:
        numpy array: The optimized cancer probability density function.
    """
    x = random.randint(0, 10000)
    randVal_gen = np.random.RandomState(x)
    peak = 1
    best = np.copy(cancer_pdf)
    if c.GOF_SMOOTHING:
        # Smooth initial solution
        modelIncid = des.run(best).cancerIncArr
        # Get integer of largest value of modelIncid
        maxVal = modelIncid.max()
        if maxVal >= 300: # if over 300 (really large peak), replace the value to better smoothing
            i_max = np.argmax(modelIncid)
            # Replace value with middle of previous and next model value
            newVal = (modelIncid[i_max - 1] + modelIncid[i_max + 1])/2
            modelIncid[i_max] = newVal
        # Smooth initial model incidence
        smoothmodelIncid = smooth_incidence(modelIncid, 21, 3) # use smoothed incidence to calculate gof
        # Remove any negative values in smoothmodelIncid
        for j in range(len(smoothmodelIncid)):
            if smoothmodelIncid[j] < 0:
                smoothmodelIncid[j] = 0
        best_eval = objective(smoothmodelIncid, min_age, max_age, cancer_inc)  # evaluate the initial point, this gives us the gof
    else:
        best = np.copy(cancer_pdf)
        best_eval = objective(des.run(best).cancerIncArr, min_age, max_age, cancer_inc)  # evaluate the initial point
    
    curr, curr_eval = best, best_eval  # current working solution

    for i in tqdm(range(n_iterations)):  # running algorithm
        if i!=0 and i%50==0:
            if peak == 0:
                mask_size = mask_size+0.1
                if mask_size>0.55:
                    peak = 1
            else:
                mask_size = mask_size-0.1
                if mask_size<0.15:
                    peak = 0
        candidate = step(np.copy(curr), step_size, mask_size, n_th_iteration = i)

        if c.GOF_SMOOTHING:
            # Smooth model incidence
            newmodelIncid = des.run(candidate).cancerIncArr
            # Get integer of largest value of modelIncid
            maxVal = newmodelIncid.max()
            if maxVal >= 300:
                i_max = np.argmax(newmodelIncid)
                # Replace value with middle of previous and next model value
                newVal = (newmodelIncid[i_max - 1] + newmodelIncid[i_max + 1])/2
                newmodelIncid[i_max] = newVal
            # Smooth new incidence
            smoothnewmodelIncid = smooth_incidence(newmodelIncid, 21, 3)
            for j in range(len(smoothnewmodelIncid)):
                if smoothnewmodelIncid[j] < 0:
                    smoothnewmodelIncid[j] = 0
            candidate_eval = objective(smoothnewmodelIncid, min_age, max_age, cancer_inc)
        else:
            candidate_eval = objective(des.run(candidate).cancerIncArr, min_age, max_age, cancer_inc)

        ## Check if candidate is changing every iteration
        # plt.plot(des.run(candidate).cancerIncArr, color = 'r')
        # plt.plot(cancer_inc, color = 'k')
        # plt.title(str(i)+'th iteration'+'_'+str(sum(candidate)))
        # plt.legend(['sim', 'SEER'])
        # plt.xlim([0,80])
        # plt.ylim([0, max(cancer_inc)+10])
        # plt.show()
        t = start_temp /(1+np.log(i+1)) # calculate temperature for current epoch
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval 
        if verbose and i%100==0:
            print(f"Iteration: {i}, Score = {best_eval}")  # report progress 
            # print('ccc:', sum(candidate))        
        diff = candidate_eval - curr_eval  # difference between candidate and current point evaluation
        metropolis = np.exp(-diff / t)  # calculate metropolis acceptance criterion
        if diff < 0 or randVal_gen.random_sample() < metropolis:  # check if we should keep the new point
            curr, curr_eval = candidate, candidate_eval  # store the new current point

            if c.GOF_SMOOTHING:
                # Limit the plot's y-axis to just above the highest SEER incidence
                plt.plot(np.arange(c.START_AGE, c.END_AGE + 1), newmodelIncid, label='Model', color='blue')
                plt.plot(np.arange(c.START_AGE, c.END_AGE + 1), smoothnewmodelIncid, label='Smooth', color='green', alpha=0.5)
                plt.legend(loc='upper left')
                plt.ylim(0, cancer_inc.max() + 50)
                plt.xlabel('Age')
                plt.ylabel('Incidence (per 100k)')
                plt.savefig(c.PATHS['plots_calibration'] + f"{c.COHORT_SEX}_{c.COHORT_RACE}_{c.CANCER_SITES[0]}_{i}.png", bbox_inches='tight')
                plt.clf()

    print(best_eval)
    return best


