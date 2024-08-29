import os
from tqdm import tqdm

# cohort_years = list(range(1935, 1961))
cohort_years = [1960]

# comb - [(sex, race, ed)]
combinations = [('Female', 'Black', ['Lung']),
 ('Female', 'Black', ['Lung', 'Colorectal']),
 ('Female', 'Black', ['Lung', 'Colorectal', 'Pancreas']),
 ('Female', 'Black', ['Lung', 'Colorectal', 'Pancreas', 'Breast']),
 ('Female', 'White', ['Lung']),
 ('Female', 'White', ['Lung', 'Colorectal']),
 ('Female', 'White', ['Lung', 'Colorectal', 'Pancreas']),
 ('Female', 'White', ['Lung', 'Colorectal', 'Pancreas', 'Breast']),
 ('Male', 'Black', ['Lung']),
 ('Male', 'Black', ['Lung', 'Colorectal']),
 ('Male', 'Black', ['Lung', 'Colorectal', 'Pancreas']),
 ('Male', 'Black', ['Lung', 'Colorectal', 'Pancreas', 'Prostate']),
 ('Male', 'White', ['Lung']),
 ('Male', 'White', ['Lung', 'Colorectal']),
 ('Male', 'White', ['Lung', 'Colorectal', 'Pancreas']),
 ('Male', 'White', ['Lung', 'Colorectal', 'Pancreas', 'Prostate'])]


# Natural history (2min)
for cohort_year in tqdm(cohort_years):
    for cohort_sex in ['Male', 'Female']:
        for cohort_race in ['White', 'Black']:
            if cohort_sex == 'Male':
                cancer_sites = ' '.join(['Lung', 'Colorectal', 'Pancreas', 'Prostate'])
            else:
                cancer_sites = ' '.join(['Lung', 'Colorectal', 'Pancreas', 'Breast'])

            # Execute the command
            os.system(f'python main.py --sojourn_time False '
                      f'--cohort_year {cohort_year} --start_age {0} '
                      f'--cohort_sex {cohort_sex} --cohort_race {cohort_race} '
                      f'--cancer_sites {cancer_sites} --cancer_sites_ed ""')

# Early detection (7min)     
for cohort_year in cohort_years:
    for comb in tqdm(combinations):
        cohort_sex = comb[0]
        cohort_race = comb[1]
        cancer_sites_ed = ' '.join(comb[2])
        if cohort_sex == 'Male':
            cancer_sites = ' '.join(['Lung', 'Colorectal', 'Pancreas', 'Prostate'])
        else:
            cancer_sites = ' '.join(['Lung', 'Colorectal', 'Pancreas', 'Breast'])   
            
        os.system(f'python main.py --sojourn_time True '
                f'--cohort_year {cohort_year} --start_age {0} '
                f'--cohort_sex {cohort_sex} --cohort_race {cohort_race} '
                f'--cancer_sites {cancer_sites} --cancer_sites_ed {cancer_sites_ed}')
            
