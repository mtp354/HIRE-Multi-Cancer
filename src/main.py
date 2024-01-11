import simpy
import random

def cancer_rate(age):
    """ Calculate the cancer rate based on age """
    # Example: cancer rate increases linearly with age
    return 0.02 + 0.001 * age

def death_rate(age):
    """ Calculate the death rate based on other causes based on age """
    # Example: death rate increases linearly with age
    return 0.05 + 0.002 * age

def life(env, name, initial_age, max_age, case, death):
    """
    Simulate an individual's life.
    :param env: SimPy environment
    :param name: Name of the individual
    :param initial_age: Initial age of the individual
    :param max_age: Maximum age for the simulation
    """
    age = initial_age
    while age < max_age:
        cr = cancer_rate(age)
        dr = death_rate(age)

        time_to_cancer = random.expovariate(cr)
        time_to_death = random.expovariate(dr)

        time_to_next_year = 1.0  # One year
        next_event = min(time_to_cancer, time_to_death, time_to_next_year)

        yield env.timeout(next_event)
        age += next_event / time_to_next_year  # Increment age

        if next_event == time_to_cancer:
            print(f"At time {env.now}, age {age}: {name} got cancer.")
            case[name]['time'] = env.now
            case[name]['age'] = age
            return  # Stop the simulation for this individual
        elif next_event == time_to_death:
            print(f"At time {env.now}, age {age}: {name} died of other causes.")
            death[name]['time'] = env.now
            death[name]['age'] = age
            return  # Stop the simulation for this individual
        # Else, one year passes without any event
    return case, death
# Simulation parameters
SIM_TIME = 100  # Total simulation time
MAX_AGE = 100   # Maximum age to simulate
num_person = 5

# Set up the environment
env = simpy.Environment()

# Create multiple individuals with different initial ages
case, death = {}, {}
for i in range(num_person):
    case[f'Person {i}'], death[f'Person {i}'] = {}, {}
    initial_age = random.randint(20, 50)  # Random initial age between 20 and 50
    env.process(life(env, f'Person {i}', initial_age, MAX_AGE, case, death))

# Run the simulation
env.run(until=SIM_TIME)

# Calculate incidence_rate and mortality_rate
def total_time(case, death):
    return sum([i['time'] for i in list(case.values()) if len(i)>0]+[i['time'] for i in list(death.values()) if len(i)>0])

incidence_rate = sum(1 for value in case.values() if value)/total_time(case, death)*100000
mortality_rate = sum(1 for value in death.values() if value)/total_time(case, death)*100000

print(incidence_rate)
print(mortality_rate)