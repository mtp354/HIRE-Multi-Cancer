# Calibration for phase1 of the model
import numpy as np
import pandas as pd
import configs as c
import phase1_model as phase1

# Set simulated annealing parameters
sim_anneal_params = {
    'starting_T': 1.0,
    'final_T': 0.1, # 0.01
    'cooling_rate': 0.3, # 0.9
    'iterations': 10} # 100

# Load all cancer incidence target data
cancer_target_df = pd.read_csv(c.INPUT_PATHS['cancer_incid'] + '1950_BC_All_Incidence.csv', index_col = 1)
target_cancer_incid = cancer_target_df['Rate'].to_numpy()

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
    init_cancer_incid = phase1.run_des(c.NUM_PATIENTS, cancer_pdf, p_cancer)

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
            new_cancer_incid = phase1.run_des(c.NUM_PATIENTS, new_cancer_pdf, new_p_cancer)
        
            # Calculate new gof
            new_gof = gof(new_cancer_incid, target_cancer_incid)
            #print("new_gof:", new_gof)
            ap =  acceptance_prob(old_gof, new_gof, T)
            #print("ap:", ap)

            # Decide if the new solution is accepted
            if np.random.uniform() < ap:
                cancer_pdf = new_cancer_pdf
                p_cancer = new_p_cancer
                old_gof = new_gof
                print(T, i, new_gof)

        T = T * sim_anneal_params['cooling_rate']
    
    return cancer_pdf, p_cancer