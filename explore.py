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