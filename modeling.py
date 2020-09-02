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

def create_knn_model(X_test_h1n1, y_test_h1n1, X_test_seas, y_test_seas):
    # fitting our data to our KNearest Neighbors model
    knn_t = KNeighborsClassifier(n_neighbors=4)
    knn_t.fit(X_test_h1n1, y_test_h1n1)

    # creating our confusion matrix
    y_pred_test = knn_t.predict(X_test_h1n1)
    pd.DataFrame(confusion_matrix(y_test_h1n1, y_pred_test))
    #code to label matrix
    #code to label matrix
    labels = ['No Vaccine', 'Vaccine']
    predicted_labels = [name + " Predicted" for name in labels ]

    h1n1_test_confusion_matrix = pd.DataFrame(confusion_matrix(y_pred_test, y_test_h1n1), index=labels, columns=[predicted_labels])
    h1n1_test_confusion_matrix.index.name = "Actual"
    h1n1_test_confusion_matrix

    # Print a classification report
    target_names = ["No Vaccine", "Vaccine"]
    print(classification_report(y_test_h1n1, y_pred_test, target_names = target_names))


    # fitting our data to our KNearest Neighbors model
    knn_st = KNeighborsClassifier(n_neighbors=4)
    knn_st.fit(X_test_seas, y_test_seas)

    # creating our confusion matrix
    y_pred_st = knn_st.predict(X_test_seas)
    pd.DataFrame(confusion_matrix(y_test_seas, y_pred_st))
    #code to label matrix
    #code to label matrix
    labels = ['No Vaccine', 'Vaccine']
    predicted_labels = [name + " Predicted" for name in labels ]

    seasonal_test_confusion_matrix = pd.DataFrame(confusion_matrix(y_pred_st, y_test_seas), index=labels, columns=[predicted_labels])
    seasonal_test_confusion_matrix.index.name = "Actual"
    seasonal_test_confusion_matrix

    # Print a classification report
    target_names = ["No Vaccine", "Vaccine"]
    print(classification_report(y_test_seas, y_pred_st, target_names = target_names))

    return h1n1_test_confusion_matrix, seasonal_test_confusion_matrix

def create_prediction_df(X_test_h1n1, y_test_h1n1, X_test_seas, y_test_seas):
    knn_t = KNeighborsClassifier(n_neighbors=4)
    knn_t.fit(X_test_h1n1, y_test_h1n1)

    knn_st = KNeighborsClassifier(n_neighbors=4)
    knn_st.fit(X_test_seas, y_test_seas)

    # Make predictions and prediction probabilities
    y_pred_proba_df = knn_t.predict_proba(X_test_h1n1)
    y_pred_proba_df_s = knn_st.predict_proba(X_test_seas)

    predictions_df = pd.DataFrame(
    {'Respondent_id': X_test_h1n1.index,
    'Probability_of_getting_h1n1_vaccine': y_pred_proba_df[:,1],
    'Probability_of_getting_seasonal_vaccine': y_pred_proba_df_s[:,1]})

    predictions_df = predictions_df.set_index('Respondent_id')
    return predictions_df

    







