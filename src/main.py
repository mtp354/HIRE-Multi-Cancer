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
        plt.plot(model.run(c.CANCER_PDF).get_incidence(), label='Model', color='blue')
        plt.scatter(x=np.arange(1975-c.COHORT_YEAR,2021-c.COHORT_YEAR), y=c.CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Type={c.COHORT_TYPE}")
        plt.show()
    elif c.MODE == 'calibrate':
        # Run calibration for simplest verion of the model
        best = simulated_annealing(model)
        # Save as numpy file, time_stamped
        np.save(c.OUTPUT_PATHS['calibration'] + f"{c.COHORT_TYPE}_{c.COHORT_YEAR}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npy", best)

        plt.plot(model.run(best).get_incidence(), label='Model', color='blue')
        plt.scatter(x=np.arange(1975-c.COHORT_YEAR,2021-c.COHORT_YEAR), y=c.CANCER_INC, label='SEER', color='darkred', alpha=0.5)
        plt.legend(loc='upper left')
        plt.xlabel('Age')
        plt.ylabel('Incidence (per 100k)')
        plt.title(f"Cancer Incidence by Age for Birthyear={c.COHORT_YEAR}, Type={c.COHORT_TYPE}")
        plt.show()

    end = timer()
    print(f'total time: {timedelta(seconds=end-start)}')

