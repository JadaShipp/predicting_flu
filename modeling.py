import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import acquire
import prepare
import explore

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
    train = explore.feature_engineering(train)
    test = explore.feature_engineering(test)

    train, validate = train_test_split(train, random_state=123, test_size=.2)
    X_train = train[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_train = train[['h1n1_vaccine']]


    X_val = validate[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_val = validate[['h1n1_vaccine']]

    X_test = test[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds']]
    y_test = test[['h1n1_vaccine']]   


    return X_train, y_train, X_val, y_val, X_test, y_test

def create_seas_validate_set(train, test):
    train = explore.feature_engineering(train)
    test = explore.feature_engineering(test)
    
    train, validate = train_test_split(train, random_state=123, test_size=.2)
    X_train = train[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds', 'doctor_recc_seasonal',
                          'opinion_seas_vacc_effective', 'opinion_seas_risk','opinion_seas_sick_from_vacc']]
    y_train = train[['seasonal_vaccine']]


    X_val = validate[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds', 'doctor_recc_seasonal',
                          'opinion_seas_vacc_effective', 'opinion_seas_risk','opinion_seas_sick_from_vacc']]
    y_val = validate[['seasonal_vaccine']]

    X_test = test[['encoded_sex', 'encoded_marital_status', 'encoded_rent_or_own', 
                          'encoded_employment_status', 'graduated_college', 'age_55_and_up', 'over_75k', 'relevant_geography',
                          'opinion_h1n1_risk_scaled', 'doctor_recc_h1n1','behavioral_antiviral_meds', 'doctor_recc_seasonal',
                          'opinion_seas_vacc_effective', 'opinion_seas_risk','opinion_seas_sick_from_vacc']]
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







