import main
import configs as c

cancers = ["Colorectal", "Pancreas", "Breast", "Prostate"]
sexes = ["Female", "Male"]
races = ["Black", "White"]


if __name__ == '__main__':
    for cancer in cancers:
        for sex in sexes:
            for race in races:
                if not ((cancer == "Breast" and sex == "Male")
                        or (cancer == "Prostate" and sex == "Female")
                        or (cancer == "Colorectal" and sex == "Female" and race == "Black")):
                    # c.CANCER_SITES = [cancer]
                    # c.COHORT_SEX = sex
                    # c.COHORT_RACE = race
                    main.main( [cancer], sex, race)