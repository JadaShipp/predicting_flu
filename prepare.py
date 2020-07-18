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
   
    train['encoded_employment_status'] = encoder.fit_transform(train['employment_status'])
    train['encoded_rent_or_own'] = encoder.fit_transform(train['rent_or_own'])
    train['encoded_marital_status'] = encoder.fit_transform(train['marital_status'])
    train['encoded_sex'] = encoder.fit_transform(train['sex'])

    test['encoded_employment_status'] = encoder.fit_transform(test['employment_status'])
    test['encoded_rent_or_own'] = encoder.fit_transform(test['rent_or_own'])
    test['encoded_marital_status'] = encoder.fit_transform(test['marital_status'])
    test['encoded_sex'] = encoder.fit_transform(test['sex'])
    
    return train, test



def ohe_age_group(train, test):
    '''
    Takes in the train and test df and one hot encodes the age_group column
    then concatenates the new encoded columns onto the origional train and test
    dataframes. Returns the transformed train and test dataframes.
    '''
    # Create encoder object
    encoder = OneHotEncoder()

    # Fit on the age_group column of the train df
    encoder.fit(train[['age_group']])

    # nice columns for display
    cols = ['age_group_' + c for c in encoder.categories_[0]]

    # Transform the column on train and test and concatenate new df onto train and test dfs
    m = encoder.transform(train[['age_group']]).todense()
    train = pd.concat([
        train,
        pd.DataFrame(m, columns=cols, index=train.index)
    ], axis=1)

    m = encoder.transform(test[['age_group']]).todense()
    test = pd.concat([
        test,
        pd.DataFrame(m, columns=cols, index=test.index)
    ], axis=1)

    return train, test




