# Calibration for phase1 of the model
import numpy as np
import pandas as pd
import configs as c
import phase1_model as phase1

# Set simulated annealing parameters
sim_anneal_params = {
    'starting_T': 1.0,
    'final_T': 0.01, # 0.01
    'cooling_rate': 0.9, # 0.9
    'iterations': 100} # 100

# Load all cancer incidence target data
cancer_target_df = pd.read_csv(c.INPUT_PATHS['cancer_incid'] + '1950_BC_All_Incidence.csv', index_col = 1)
target_cancer_incid = cancer_target_df['Rate'].to_numpy()

# Function to calculate model incidence
def get_incidence(cancerArr, acMortArr):
    # Get number of patients who are alive at the start of each year
    live = np.zeros(len(range(c.START_AGE, c.END_AGE)))
    for i in range(len(live)):
        if i == 0:
            live[i] = c.NUM_PATIENTS - acMortArr[i]
        else:
            live[i] = live[i - 1] - acMortArr[i]
    # Restrict live and cancer Arr to the cancer incidence ages
    live = live[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]
    cancerArr = cancerArr[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]

    # Get cancer incidence for each age
    cancer_incid = cancerArr / live * 100_000
    # Only from ages 25 - 70 to align with target data
    cancer_incid = cancer_incid[c.ALL_AGES.index(25):c.ALL_AGES.index(70) + 1]
    return cancer_incid

# Goodness-of-fit functions
def gof(obs, exp):
    # chi-squared
    # inputs: numpy arrays of observed and expected values
    chi = ((obs-exp)**2)
    chi_sq = sum(chi)
    return chi_sq

# Functions for running simulated annealing algorithm
def select_new_params(step, old_param):
    '''Selects new param within range old_param +/- step%
       step: proportion to change param (between 0 and 1), does not depend on temperature
       old_param: old parameter
       Outputs a new parameter'''
    new_param = np.random.uniform(old_param - old_param * step, old_param + old_param * step)
    while new_param < 0:
        new_param = np.random.uniform(old_param - old_param * step, old_param + old_param * step)
    return new_param

def generate_cancer_pdf(curr_cancer_pdf):
    new_pdf = curr_cancer_pdf.copy()
    for i in range(len(new_pdf)):
        new_pdf[i] = select_new_params(0.4, new_pdf[i])
    
    new_pdf /= new_pdf.sum()

    return new_pdf

def acceptance_prob(old_gof, new_gof, T):
    if new_gof < old_gof:
        return 1
    else:
        return np.exp((old_gof - new_gof) / T)

# Simulated annealing algorithm
def anneal(init_cancer_pdf, init_p_cancer):
    # Get first solution for initial cdf
    cancer_pdf = init_cancer_pdf
    p_cancer = init_p_cancer
    cancerArr, acMortArr = phase1.run_des(c.NUM_PATIENTS, cancer_pdf, p_cancer)

    # Calculate incidence
    init_cancer_incid = get_incidence(cancerArr, acMortArr)

    # Calculate gof
    old_gof = gof(init_cancer_incid, target_cancer_incid)
    print("old_gof:", old_gof)

    # Starting temperature
    T = sim_anneal_params['starting_T']

    # Start temperature loop
    # Annealing schedule
    while T > sim_anneal_params['final_T']:
        # Sampling at T
        for i in range(sim_anneal_params['iterations']):
            # Find new candidate parameters
            new_cancer_pdf = generate_cancer_pdf(cancer_pdf)
            new_p_cancer = select_new_params(0.4, p_cancer)

            # Get new solutions
            new_cancerArr, new_acMortArr = phase1.run_des(c.NUM_PATIENTS, new_cancer_pdf, new_p_cancer)
            new_cancer_incid = get_incidence(new_cancerArr, new_acMortArr)
        
            # Calculate new gof
            new_gof = gof(new_cancer_incid, target_cancer_incid)
            print("new_gof:", new_gof)
            ap =  acceptance_prob(old_gof, new_gof, T)
            print("ap:", ap)

            # Decide if the new solution is accepted
            if np.random.uniform() < ap:
                cancer_pdf = new_cancer_pdf
                p_cancer = new_p_cancer
                old_gof = new_gof
                print(T, i, new_gof)
                # Just in case calibration fails, save each best parameter
                # Save as numpy file
                np.save(c.OUTPUT_PATHS['calibration'] + 'cancer_pdf_01262024', cancer_pdf)
                p_cancer_df = pd.DataFrame(np.array([p_cancer]), columns = ['prob'])
                p_cancer_df.to_excel(c.OUTPUT_PATHS['calibration'] + 'p_cancer_01262024.xlsx')

        T = T * sim_anneal_params['cooling_rate']
    
    return cancer_pdf, p_cancer