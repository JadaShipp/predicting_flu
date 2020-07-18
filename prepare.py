import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# ---------------------#
#  Initial Evaluation  #
#      Functions       #
# ---------------------#

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

# ---------------------#
#  Prepare functions   #
# ---------------------#


def fill_null_values(train, test):
    train = train.apply(lambda x:x.fillna(x.value_counts().index[0]))
    test = test.apply(lambda x:x.fillna(x.value_counts().index[0]))

    return train, test

# ---------------------#
#    Label Encoding    #
# ---------------------#

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

# ---------------------#
#   One Hot Encoding   #
# ---------------------#

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



def ohe_education(train, test):
    # Encode education column

    # Create encoder object
    encoder = OneHotEncoder()

    # Fit on the age_group column of the train df
    encoder.fit(train[['education']])

    # nice columns for display
    cols = ['education_' + c for c in encoder.categories_[0]]

    # Transform the column on train and test and concatenate new df onto train and test dfs
    m = encoder.transform(train[['education']]).todense()
    train = pd.concat([
        train,
        pd.DataFrame(m, columns=cols, index=train.index)
    ], axis=1)

    m = encoder.transform(test[['education']]).todense()
    test = pd.concat([
        test,
        pd.DataFrame(m, columns=cols, index=test.index)
    ], axis=1)

    return train, test

def ohe_race(train, test):
    # Encode race column

    # Create encoder object
    encoder = OneHotEncoder()

    # Fit on the age_group column of the train df
    encoder.fit(train[['race']])

    # nice columns for display
    cols = ['race_' + c for c in encoder.categories_[0]]

    # Transform the column on train and test and concatenate new df onto train and test dfs
    m = encoder.transform(train[['race']]).todense()
    train = pd.concat([
        train,
        pd.DataFrame(m, columns=cols, index=train.index)
    ], axis=1)

    m = encoder.transform(test[['race']]).todense()
    test = pd.concat([
        test,
        pd.DataFrame(m, columns=cols, index=test.index)
    ], axis=1)

    return train, test

def ohe_income_poverty(train,test):
    # Encode income_poverty column

    # Create encoder object
    encoder = OneHotEncoder()

    # Fit on the age_group column of the train df
    encoder.fit(train[['income_poverty']])

    # nice columns for display
    cols = ['income_poverty_' + c for c in encoder.categories_[0]]

    # Transform the column on train and test and concatenate new df onto train and test dfs
    m = encoder.transform(train[['income_poverty']]).todense()
    train = pd.concat([
        train,
        pd.DataFrame(m, columns=cols, index=train.index)
    ], axis=1)

    m = encoder.transform(test[['income_poverty']]).todense()
    test = pd.concat([
        test,
        pd.DataFrame(m, columns=cols, index=test.index)
    ], axis=1)

    return train, test

# ---------------------#
#        Scaling       #
# ---------------------#

def scale_minmax(train, test, column_list):
    scaler = MinMaxScaler()
    column_list_scaled = [col + '_scaled' for col in column_list]
    train_scaled = pd.DataFrame(scaler.fit_transform(train[column_list]), 
                                columns = column_list_scaled, 
                                index = train.index)
    train = train.join(train_scaled)

    test_scaled = pd.DataFrame(scaler.transform(test[column_list]), 
                                columns = column_list_scaled, 
                                index = test.index)
    test = test.join(test_scaled)

    return train, test