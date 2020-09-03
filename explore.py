import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import acquire
import prepare

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

#Set figure size and figure size for all plots
plt.rc("figure", figsize = (14,14))
plt.rc("font", size=14)

# Allow all columns to be displayed
pd.set_option('display.max_columns', None)

# ---------------------#
#     H1N1 Explore     #
# ---------------------#



def distribution_of_h1n1_vaccine_status(h1n1_train):
    '''
    Takes in the train set and returns a barplot depicting
    number of people vaccinated and not vaccinated.
    '''
    #Plot the distribution on of vaccine status
    plt.figure(figsize=(15,15))
    h1n1_train.h1n1_vaccine.value_counts().sort_index().plot(kind = "bar",alpha = .5)
    plt.title("Distirbution of Patients' Vaccine Status")

def sex_race_education_h1n1(h1n1_train):
    features = ['sex', 'race', 'education']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = h1n1_train.h1n1_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'h1n1_vaccine', data=h1n1_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate')
        ax[i].axhline(vaccination_rate, ls='--', color='grey')  

def age_income_marital_status(h1n1_train):
    features = ['age_group', 'income_poverty', 'marital_status']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = h1n1_train.h1n1_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'h1n1_vaccine', data=h1n1_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate')
        ax[i].axhline(vaccination_rate, ls='--', color='grey')

def rent_employment_geo(h1n1_train):
    features = ['rent_or_own', 'employment_status', 'hhs_geo_region']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = h1n1_train.h1n1_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'h1n1_vaccine', data=h1n1_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate', fontsize = 16.0)
        ax[i].axhline(vaccination_rate, ls='--', color='grey')


def chi_square_opinion_status(h1n1_train):
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


def opinion_h1n1_status(h1n1_train):
    plt.rc('figure', figsize=(15, 15))
    plt.rc('font', size=13)
    plt.title('Opinion of H1N1 Risk and Vaccine Status')
    ctab = pd.crosstab(h1n1_train.h1n1_vaccine, h1n1_train.opinion_h1n1_risk, normalize=True)
    sns.heatmap(ctab, annot=True, cmap='Purples', fmt='.2%')



def create_continuous_heatmap_h1n1(h1n1_train):
    # Subset data to only have opinion
    continuous_h1n1 = h1n1_train[['h1n1_concern',
                                'h1n1_knowledge',
                                'behavioral_antiviral_meds',
                                'behavioral_avoidance',
                                'behavioral_face_mask',
                                'behavioral_wash_hands',
                                'behavioral_large_gatherings',
                                'behavioral_outside_home',
                                'behavioral_touch_face',
                                'doctor_recc_h1n1',
                                'chronic_med_condition',
                                'child_under_6_months',
                                'health_worker',
                                'health_insurance',
                                'opinion_h1n1_vacc_effective',
                                'opinion_h1n1_risk',
                                'opinion_h1n1_sick_from_vacc',
                                'h1n1_vaccine']]
    plt.matshow(continuous_h1n1.corr())
    plt.xticks(range(len(continuous_h1n1.columns)), continuous_h1n1.columns)
    plt.yticks(range(len(continuous_h1n1.columns)), continuous_h1n1.columns)
    plt.colorbar()
    plt.figure(figsize = (10.0, 10.0))
    plt.show()
    
def h1n1_doc_recc_correlation(h1n1_train):
    x = h1n1_train.doctor_recc_h1n1
    y = h1n1_train.h1n1_vaccine

    corr, p = stats.pearsonr(x, y)
    corr, p

    alpha = .05

    if p < alpha:
        print(f'''
        Because p ({p:.4f}) is less than alpha ({alpha:.2f}), Based on the pearson's r test, 
        it looks like there's a fairly strong correlation between the h1n1 vaccine status and
        whether the doctor recommends getting it. 
        ''')
    else:
        print(f'''Because p ({p:.4f}) is greater than alpha ({alpha:.2f}),  Based on the pearson's r test, 
        it looks like there is no strong correlation between the h1n1 vaccine status and
        whether the doctor recommends getting it. 
        ''')


def feature_engineering(train):
    train['graduated_college'] = train['education_college_graduate'] > 0
    train['age_55_and_up'] = train[['age_group_55_-_64_years', 'age_group_65+_years']].sum(axis=1) > 0
    train['over_75k'] = train['income_poverty_>_$75,000'] > 0
    train['relevant_geography'] = train[['hhs_geo_region_bhuqouqj', 'hhs_geo_region_dqpwygqj', 
                                     'hhs_geo_region_fpwskwrf', 'hhs_geo_region_kbazzjca', 
                                     'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit']].sum(axis = 1)> 0
    return train



# ---------------------#
#   Seasonal Explore   #
# ---------------------#

def distribution_of_seasonal_vaccine_status(seasonal_train):
    '''
    Takes in the train set and returns a barplot depicting
    number of people vaccinated and not vaccinated.
    '''
   #Plot the distribution on of vaccine status
    plt.figure(figsize=(15,15))
    seasonal_train.seasonal_vaccine.value_counts().sort_index().plot(kind = "bar",alpha = .5)
    plt.title("Distirbution of Patients' Vaccine Status")



def opinion_seasonal_status(train):
    #Create a function that takes in the h1n1 training set and creates a heatmaps
    plt.rc('figure', figsize=(15, 15))
    plt.rc('font', size=13)
    plt.title('Opinion of Seasoonal Risk and Vaccine Status')
    ctab = pd.crosstab(train.seasonal_vaccine, train.opinion_seas_risk, normalize=True)
    sns.heatmap(ctab, annot=True, cmap='Purples', fmt='.1%')

def sex_race_education_seasonal(seasonal_train):
    features = ['sex', 'race', 'education']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = seasonal_train.seasonal_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'seasonal_vaccine', data=seasonal_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate')
        ax[i].axhline(vaccination_rate, ls='--', color='grey')
  

def age_income_marital_status_seaonal(seasonal_train):
    features = ['age_group', 'income_poverty', 'marital_status']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = seasonal_train.seasonal_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'seasonal_vaccine', data=seasonal_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate')
        ax[i].axhline(vaccination_rate, ls='--', color='grey')

def rent_employment_geo_seaonal(seasonal_train):
    features = ['rent_or_own', 'employment_status', 'hhs_geo_region']

    _, ax = plt.subplots(nrows=1, ncols=3, figsize=(25,25))

    vaccination_rate = seasonal_train.seasonal_vaccine.mean()

    for i, feature in enumerate(features):
        sns.barplot(feature, 'seasonal_vaccine', data=seasonal_train, ax=ax[i], alpha=.9)
        ax[i].set_ylabel('Vaccination Rate', fontsize = 16.0)
        ax[i].axhline(vaccination_rate, ls='--', color='grey')


def create_continuous_heatmap_seasonal(seasonal_train):
    # Subset data to only have opinion
    continuous_seasonal = seasonal_train[[
                                'behavioral_antiviral_meds',
                                'behavioral_avoidance',
                                'behavioral_face_mask',
                                'behavioral_wash_hands',
                                'behavioral_large_gatherings',
                                'behavioral_outside_home',
                                'behavioral_touch_face',
                                'doctor_recc_seasonal',
                                'chronic_med_condition',
                                'child_under_6_months',
                                'health_worker',
                                'health_insurance',
                                'opinion_seas_vacc_effective',
                                'opinion_seas_risk',
                                'opinion_seas_sick_from_vacc',
                                'seasonal_vaccine']]
    plt.matshow(continuous_seasonal.corr())
    plt.xticks(range(len(continuous_seasonal.columns)), continuous_seasonal.columns)
    plt.yticks(range(len(continuous_seasonal.columns)), continuous_seasonal.columns)
    plt.colorbar()
    plt.figure(figsize = (10.0, 10.0))
    plt.show()