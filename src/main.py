# A super simple discrete event simulation model based on the simpy package.

import configs as c
from timeit import default_timer as timer
from datetime import timedelta
import phase1_model as phase1

if __name__ == '__main__':
    if c.MODE == 'phase1':
        # Run simplest verion of the model
        start = timer()
        phase1.run_phase1()
        end = timer()
        print(f'total time: {timedelta(seconds=end-start)}')