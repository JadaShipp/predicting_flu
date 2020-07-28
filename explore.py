import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import acquire
import prepare

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Set figure size and figure size for all plots
plt.rc("figure", figsize = (14,14))
plt.rc("font", size=14)

# Allow all columns to be displayed
pd.set_option('display.max_columns', None)


def opinion_h1n1_status(train):
    plt.title('Opinion of H1N1 Risk and Vaccine Status')
    ctab = pd.crosstab(h1n1_train.h1n1_vaccine, h1n1_train.opinion_h1n1_risk, normalize=True)
    sns.heatmap(ctab, annot=True, cmap='Purples', fmt='.2%')

def chi_square_opinion_status(train):
    observed = pd.crosstab(h1n1_train.h1n1_vaccine, h1n1_train.opinion_h1n1_risk)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}') 

    alpha = .05

    if p < alpha:
    print(f'''
        Because p ({p:.4f}) is less than alpha ({alpha:.2f}), we reject the null hypothesis.
        This means that the chances of observing the relationship between 
        opinion and vaccine status due to chance are slim. 
        ''')
    else:
    print(f'''Because p ({p:.4f}) is greater than alpha ({alpha:.2f}), we fail to reject the null hypothesis.
        This means there is a statistically significant probability of observing the relationship between 
        opinion and vaccine status due to chance. 
        ''')  

