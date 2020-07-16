import pandas as import pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



def percent_nans(df):
    '''
    Takes in a dataframe and returns a dataframe with the column name, 
    the number of nans in that column and the percent of nans in that column
    '''
    x = ['column_name','n_nans', 'percentage_nans']
    missing_data = pd.DataFrame(columns=x)
    columns = df.columns
    for col in columns:
        acolumn_name = col
        amissing_data = df[col].isnull().sum()
        amissing_in_percentage = (df[col].isnull().sum()/df[col].shape[0])*100
        
        missing_data.loc[len(missing_data)] = [acolumn_name, amissing_data, amissing_in_percentage]
    return missing_data


def fill_null_values(train, test):
    train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
    test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))

    return train, test

def label_encode_columns(train, test):

    encoder = LabelEncoder()
   
    encode_list = ['rent_or_own', 'employment_status', 'marital_status', 'sex']
    
             
    for e in encode_list:
        train[e] = encoder.fit_transform(train[e])
        test[e] = encoder.transform(test[e])

        return train, test

def one_hot_encode_columns(train, test):
    encoder = OneHotEncoder()

    encode_list = ['age_group', 'education', 'race', 'income_poverty']

    for e in encode_list:
        train[e] = encoder.fit_transform(train[e])
        test[e] = encoder.transform(test[e])

        return train, test


