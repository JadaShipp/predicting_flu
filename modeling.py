import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import acquire
import prepare

# Modeling

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def create_h1n1_validate_set(train, test):
    train, validate = train_test_split(train, random_state=123, test_size=.2)
    X_train = train[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_train = train[['h1n1_vaccine']]


    X_val = validate[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_val = validate[['h1n1_vaccine']]

    X_test = test[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_test = test[['h1n1_vaccine']]   


    return X_train, y_train, X_val, y_val, X_test, y_test

def create_seas_validate_set(train, test):
    train, validate = train_test_split(train, random_state=123, test_size=.2)
    X_train = train[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_train = train[['seasonal_vaccine']]


    X_val = validate[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_val = validate[['seasonal_vaccine']]

    X_test = test[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'race_black', 'race_hispanic', 
                          'race_other_or_multiple', 'race_white','education_12_years', 'education_<_12_years',
                          'education_college_graduate', 'education_some_college', 'age_group_18_-_34_years', 'age_group_35_-_44_years',
                          'age_group_45_-_54_years', 'age_group_55_-_64_years', 'age_group_65+_years', 
                          'income_poverty_<=_$75,000,_above_poverty','income_poverty_>_$75,000','income_poverty_below_poverty',
                          'hhs_geo_region_atmpeygn', 'hhs_geo_region_bhuqouqj','hhs_geo_region_dqpwygqj', 'hhs_geo_region_fpwskwrf',
                          'hhs_geo_region_kbazzjca', 'hhs_geo_region_lrircsnp', 'hhs_geo_region_lzgpxyit', 'hhs_geo_region_mlyzmhmf', 
                          'hhs_geo_region_oxchjgsf', 'hhs_geo_region_qufhixun', 'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_test = test[['seasonal_vaccine']]   


    return X_train, y_train, X_val, y_val, X_test, y_test


def logit_regression_model(x_train, y_train):
    # create and fit a logistic regression model
    logit = LogisticRegression(random_state = 123)
    logit.fit(X_train, y_train)

    # creating the confusion matrix
    y_pred = logit.predict(X_train)
    pd.DataFrame(confusion_matrix(y_train, y_pred))

    #code to label matrix
    labels = ['No Vaccine', 'Vaccine']
    predicted_labels = [name + " Predicted" for name in labels ]

    conf = pd.DataFrame(confusion_matrix(y_pred, y_train), index=labels, columns=[predicted_labels])
    conf.index.name = "Actual"
    
    return conf







