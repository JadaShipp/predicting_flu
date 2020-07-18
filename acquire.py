import pandas as import pd
import numpy as np
from sklearn.model_selection import train_test_split


def acquire_training_data():
    '''
    Reads in feature csv data from folder and converts it to pandas dataframe
    Returns dataframe
    '''
    # Read in the training set feature csv using pandas
    df = pd.read_csv('Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Features.csv', 
    index_col=0)

    return df

def acquire_target_variable_data():
    '''
    Reads in target variable csv data from folder and converts it to pandas dataframe
    Returns dataframe
    '''
    # Read in the training set target variable csv using pandas
    df = pd.read_csv('Flu_Shot_Learning_Predict_H1N1_and_Seasonal_Flu_Vaccines_-_Training_Labels.csv', 
    index_col=0)

    return df

def acquire_data():
    '''
    Merges feature dataframe and target variable dataframe,
    Drops duplicate respondent_id column
    Returns dataframe
    '''
    feature_df = acquire_training_data()
    target_variable_df = acquire_target_variable_data()

    df = pd.concat([df, target_variable_df], axis = 1)
    df = df.drop(columns = 'respondent_id')

    return df