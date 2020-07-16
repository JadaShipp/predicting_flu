import pandas as import pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



def label_encode_columns(df):

    encoder = LabelEncoder()

    encode_list = [''

    ]
             
    for e in encode_list:
        train[e] = encoder.fit_transform(train[e])
        test[e] = encoder.transform(test[e])

def one_hot_encode_columns(df):
    encoder = OneHotEncoder()

    encode_list = [
        ''
    ]

def percent_nans(df):
    '''
    Takes in a dataframe 
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