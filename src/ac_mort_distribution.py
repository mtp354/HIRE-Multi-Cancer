import numpy as np
import pandas as pd
import configs as c
import matplotlib.pyplot as plt 

# Code based on:
# https://people.reed.edu/~jones/141/Life.html

SAVE_PLOTS = True

# Get proportion of people who died -> cumulative distribution function (CDF)
cdf = 1 - c.life_table['Start_Num']/100000
# Plot CDF
if SAVE_PLOTS:
    fig = plt.figure(figsize = (10,5))
    plt.bar(c.life_table.index, cdf)
    plt.xlabel("Age")
    plt.ylabel("CDF")
    plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'full_acmort_cdf.png')

# Get the probability distribution function (PDF) = CDF(n) - CDF(n-1)
pdf = np.diff(np.array(cdf)) # sums to 1
# Plot PDF
if SAVE_PLOTS:
    fig = plt.figure(figsize = (10,5))
    plt.bar(c.life_table.index[1:], pdf) # doesn't start at age 0, starts at age 1
    plt.xlabel("Age")
    plt.ylabel("PDF")
    plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'full_acmort_pdf.png')

# Conditional probability based on specified start age
START_AGE = 50
# Get the conditional PDF
condProb = pdf[START_AGE:]
condPDF = condProb / sum(condProb) # sums to 1
# Plot conditional PDF
if SAVE_PLOTS:
    fig = plt.figure(figsize = (10,5))
    plt.bar(range(START_AGE + 1, 101), condPDF)
    plt.xlabel("Age")
    plt.ylabel("Conditional PDF")
    plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'cond' + str(START_AGE) + '_acmort_pdf.png')

# Get conditional CDF
condCDF = np.cumsum(condPDF)
# Plot CDF
if SAVE_PLOTS:
    fig = plt.figure(figsize = (10,5))
    plt.bar(range(START_AGE + 1, 101), condCDF)
    plt.xlabel("Age")
    plt.ylabel("Conditional CDF")
    plt.savefig(c.OUTPUT_PATHS['ac_mort_plots'] + 'cond' + str(START_AGE) + '_acmort_cdf.png')
