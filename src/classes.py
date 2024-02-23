# Defines all classes and functions used in the simulation
import numpy as np
import configs as c
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter


class Patient:
    def __init__(self, pid, starting_age=c.START_AGE):
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
        self.current_state = 'Healthy'
        self.history = {self.current_state:self.age}  # A dictionary to store the state and the age at entry to the state
    
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
    def run(self, cancer_pdf):
        self.reset()
        while 'Death' not in self.current_state:
            if self.current_state == 'Healthy':
                condCDF = np.cumsum(c.ac_pdf[int(self.age):])  # Get the conditional PDF
                time_to_od = np.searchsorted(condCDF/condCDF[-1], np.random.rand())
                
                time_to_cancer_onset = np.searchsorted(np.cumsum(cancer_pdf), np.random.rand())
                sample_sojourn_time = np.random.normal(c.MEAN_CANCER_SOJORN_TIME, c.STD_DEV_SOJOURN_TIME)
                time_to_cancer_detectable = time_to_cancer_onset + sample_sojourn_time
                
                if time_to_cancer_detectable <= time_to_od:  # If cancer happens before death
                    self.current_state = 'Cancer'
                    self.age += time_to_cancer_detectable
                else:
                    self.current_state = 'Other Death'
                    self.age += time_to_od
                self.history[self.current_state] = self.age

            if self.current_state == 'Cancer':
                time_at_risk = min(10, c.END_AGE-int(self.age)-1)
                time_to_cd = np.searchsorted(c.cancer_surv_arr[int(self.age) - c.START_AGE, :1+time_at_risk, 0], np.random.rand())
                time_to_od = np.searchsorted(c.cancer_surv_arr[int(self.age) - c.START_AGE, :1+time_at_risk, 1], np.random.rand())
                if time_to_od < time_to_cd:  # # If other death happens before cancer
                    self.current_state = 'Other Death'
                    self.age += time_to_od
                elif time_to_cd < time_at_risk:  # If cancer death happens before other death
                    self.current_state = 'Cancer Death'
                    self.age += time_to_cd
                else:
                    self.current_state = 'Healthy'
                    self.age += time_at_risk
                self.history[self.current_state] = self.age
        return self.history

    def run_intervention(self, cancer_pdf):
        self.reset()
        while 'Death' not in self.current_state:
            if self.current_state == 'Healthy':
                condCDF = np.cumsum(c.ac_pdf[int(self.age):])  # Get the conditional PDF
                time_to_od = np.searchsorted(condCDF/condCDF[-1], np.random.rand())
                
                time_to_cancer_onset = np.searchsorted(np.cumsum(cancer_pdf), np.random.rand())
                sample_sojourn_time = np.random.normal(c.MEAN_CANCER_SOJORN_TIME, c.STD_DEV_SOJOURN_TIME)
                time_to_cancer_detectable = time_to_cancer_onset + sample_sojourn_time
                
                if time_to_cancer_detectable <= time_to_od:  # If cancer detected before death
                    self.current_state = 'Cancer'
                    self.age += time_to_cancer_detectable
                else:
                    if time_to_cancer_onset <= time_to_od: # If cancer onset before death
                        if c.SCREENING_AGE <= self.age + time_to_od: #if screening happens during sojorun time
                            self.current_state = 'Cancer'
                            self.age = c.SCREENING_AGE
                        else:
                            self.current_state = 'Other Death'
                            self.age += time_to_od                      
                    else:
                        self.current_state = 'Other Death'
                        self.age += time_to_od
                self.history[self.current_state] = self.age

            if self.current_state == 'Cancer':
                time_at_risk = min(10, c.END_AGE-int(self.age)-1)
                time_to_cd = np.searchsorted(c.cancer_surv_arr[int(self.age) - c.START_AGE, :1+time_at_risk, 0], np.random.rand())
                time_to_od = np.searchsorted(c.cancer_surv_arr[int(self.age) - c.START_AGE, :1+time_at_risk, 1], np.random.rand())
                if time_to_od < time_to_cd:  # # If other death happens before cancer
                    self.current_state = 'Other Death'
                    self.age += time_to_od
                elif time_to_cd < time_at_risk:  # If cancer death happens before other death
                    self.current_state = 'Cancer Death'
                    self.age += time_to_cd
                else:
                    self.current_state = 'Healthy'
                    self.age += time_at_risk
                self.history[self.current_state] = self.age
        return self.history

    def reset(self):
        """
        Reset the state of the object by setting history to contain the current age and 0, and 
        setting karma to a random value between 0 and 1.
        """
        self.age = c.START_AGE
        self.current_state = 'Healthy'
        self.history = {self.current_state:self.age}



class DiscreteEventSimulation:
    def __init__(self, num_patients=c.NUM_PATIENTS, starting_age=c.START_AGE):
        """
        Initializes the object with the given `cancer_cdf`.
        """
        self.num_patients = num_patients
        self.patients = [Patient(pid, starting_age) for pid in range(self.num_patients)]
        self.log = []  # A log of all patient dictionaries
        self.cancerIncArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize incidence array
        self.acMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize mortality array
        self.cancerMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize cancer mortality array
    
    def run(self, cancer_pdf):
        """
        Runs the discrete event simulation for the given number of patients.
        """
        self.reset()
        for patient in self.patients:
            if c.MODE == 'intervention':
                patient_history = patient.run_intervention(cancer_pdf)  # running patient
            else:
                patient_history = patient.run(cancer_pdf)  # running patient
            self.log.append(patient_history)  # recording to log
            try:
                self.cancerIncArr[int(patient_history['Cancer']) - c.START_AGE] += 1  # Increment the incidence count for the corresponding age
            except KeyError: 
                pass
            try:
                self.acMortArr[int(patient_history['Other Death']) - c.START_AGE] += 1  # Increment the mortlity count for the corresponding age
            except KeyError: 
                pass
            try:
                self.cancerMortArr[int(patient_history['Cancer Death']) - c.START_AGE] += 1  # Increment the cancer mortality count for the corresponding age
            except KeyError: 
                pass
        num_alive = self.num_patients - self.acMortArr.cumsum() - self.cancerMortArr.cumsum()  # Adjusting denominator based on number alive
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
def objective(obs, exp=c.CANCER_INC):
        """
        Calculate the sum of the squared differences between the observed and expected values.

        Parameters:
            obs (array): The array of observed values.
            exp (array): The array of expected values.

        Returns:
            float: The sum of the squared differences.
        """
        # return np.sum(np.square(obs[1975-c.COHORT_YEAR:2021-c.COHORT_YEAR] - exp))  # Only compares the years we have incidence data
        return mean_squared_error(exp, obs[1975-c.COHORT_YEAR:2021-c.COHORT_YEAR])

def step(candidate, step_size=c.STEP_SIZE, mask_size=c.MASK_SIZE):
    """
    Generate a new candidate by adding random noise to the input candidate array, and then clipping the values to be within the range of 0.0 and 1.0.
    Parameters:
    - candidate: The input array of values.
    - step_size: The size of the random noise to be added to the candidate array.
    Returns:
    - The new candidate array with values clipped between 0.0 and 1.0.
    """
    mask = np.random.random(candidate.shape) > mask_size # fraction of values to modify
    candidate[mask] += np.random.uniform(-step_size, step_size, mask.sum())
    candidate = savgol_filter(candidate, 10, 3, mode='interp')  # smoothing
    return np.clip(candidate, 0.0, 1.0)

def simulated_annealing(des, cancer_pdf=c.CANCER_PDF, cancer_inc=c.CANCER_INC, n_iterations=c.NUM_ITERATIONS, 
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
    best = np.copy(cancer_pdf)
    best_eval = objective(des.run(best).cancerIncArr, cancer_inc)  # evaluate the initial point
    curr, curr_eval = best, best_eval  # current working solution
    for i in range(n_iterations):  # running algorithm
        candidate = step(np.copy(curr), step_size, mask_size)
        candidate_eval = objective(des.run(candidate).cancerIncArr, cancer_inc)
        t = start_temp /(1+np.log(i+1)) # calculate temperature for current epoch
        if candidate_eval < best_eval:
            best, best_eval = candidate, candidate_eval 
        if verbose and i%100==0:
            print(f"Iteration: {i}, Score = {best_eval}")  # report progress         
        diff = candidate_eval - curr_eval  # difference between candidate and current point evaluation
        metropolis = np.exp(-diff / t)  # calculate metropolis acceptance criterion
        if diff < 0 or np.random.random() < metropolis:  # check if we should keep the new point
            curr, curr_eval = candidate, candidate_eval  # store the new current point
    print(best_eval)
    return best


