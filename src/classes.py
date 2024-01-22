# Defines all classes and functions used in the simulation
import numpy as np
import configs as c


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
        self.fate = np.random.rand()  # The random value used for outcome determination
        self.history = {'Healthy':self.age}  # A dictionary to store the state and the age at entry to the state
    
    def __repr__(self) -> str:
        """
        Return a string representation of the Patient object for debugging purposes.
        This function does not take any parameters.
        It returns a string.
        """
        return f"Patient(pid={self.pid}, age={self.age}, fate={self.fate}, history={self.history})"

    def __str__(self) -> str:
        """
        Return a simplified string representation of the Patient object's attributes.
        """
        return f"Patient:{self.pid}, history:{self.history})"

    def run(self, cancer_pdf):
        """
        Runs the discrete event simulation for one patient, resets at the start for precaution.
        """
        self.reset()
        condProb = c.ac_pdf[self.age:]  # Get the conditional PDF
        condCDF = np.cumsum(condProb / sum(condProb))  # Get conditional CDF
        ac_age = np.searchsorted(condCDF, self.fate) + self.age  # Determine age of death

        cancer_cdf = np.cumsum(cancer_pdf)  # Generate CDF
        cancer_age = np.searchsorted(cancer_cdf, self.fate) + self.age  # Determine age of cancer
        if cancer_age <= ac_age:  # If cancer happens before death
            self.history['Cancer'] = cancer_age  # Add to history

        self.history['Other Death'] = ac_age  # Add to history
        return self.history
    
    def reset(self):
        """
        Reset the state of the object by setting history to contain the current age and 0, and 
        setting karma to a random value between 0 and 1.
        """
        self.history = {'Healthy':self.age}
        self.fate = np.random.rand()

class DiscreteEventSimulation:
    def __init__(self, num_patients=c.NUM_PATIENTS, starting_age=c.START_AGE):
        """
        Initializes the object with the given `cancer_pdf`.
        """
        self.num_patients = num_patients
        self.patients = [Patient(pid, starting_age) for pid in range(self.num_patients)]
        self.log = []  # A log of all patient dictionaries
    
    def run(self, cancer_pdf):
        """
        Runs the discrete event simulation for the given number of patients.
        """
        self.reset()
        for patient in self.patients:
            self.log.append(patient.run(cancer_pdf))  # running patient and recording log
        return self
    
    def reset(self):
        """
        Reset the log.
        """
        self.log = []

    def get_mortality(self):
        """
        Convert the log of all events into an age-specific mortality array
        """
        # Convert the log of all events into an age-specific mortality array
        acMortArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize mortality array
        for patient_history in self.log:
            acMortArr[patient_history['Other Death'] - c.START_AGE] += 1  # Increment the mortality count for the corresponding age
        return acMortArr

    def get_incidence(self):
        """
        Convert the log of all events into an age-specific incidence array.
        """
        # Convert the log of all events into an age-specific incidence array
        cancerIncArr = np.zeros((c.END_AGE - c.START_AGE + 1))  # Initialize incidence array
        try:
            for patient_history in self.log:
                cancerIncArr[patient_history['Cancer'] - c.START_AGE] += 1  # Increment the incidence count for the corresponding age
        except KeyError:  # if the patient never gets cancer, skip
            pass
        # Adjusting denominator based on number alive
        num_alive = self.num_patients - self.get_mortality().cumsum()

        return 100000*np.divide(cancerIncArr, num_alive+0.001)  # adding 0.001 to avoid NaNs


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
        return np.sum(np.square(obs[1975-c.COHORT_YEAR:2021-c.COHORT_YEAR] - exp))  # Only compares the years we have incidence data for

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
    return np.clip(candidate, 0.0, 1.0)

def simulated_annealing(dse, cancer_pdf=c.CANCER_PDF, n_iterations=c.NUM_ITERATIONS, step_size=c.STEP_SIZE, 
                        mask_size=c.MASK_SIZE, verbose=c.VERBOSE):
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
    best_eval = objective(dse.run(best).get_incidence(), cancer_pdf)  # evaluate the initial point
    curr, curr_eval = best, best_eval  # current working solution
    for i in range(n_iterations):  # running algorithm
        candidate = step(np.copy(curr), step_size, mask_size)
        candidate_eval = objective(dse.run(candidate).get_incidence(), cancer_pdf)
        t = 10 /(1+np.log(i+1)) # calculate temperature for current epoch
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






