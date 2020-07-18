import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ---------------------#
#  Initial Evaluation  #
#      Functions       #
# ---------------------#

def percent_nans(df):
    '''
    Takes in a dataframe and returns a dataframe with the column name, 
    the number of nans in that column and the percent of nans in that column
    '''
    x = ['column','n_nans', 'percentage_nans']
    missing_data_df = pd.DataFrame(columns=x)
    columns = df.columns
    for col in columns:
        column_name = col
        missing_data = df[col].isnull().sum()
        missing_in_percentage = (df[col].isnull().sum()/df[col].shape[0])*100
        
        missing_data_df.loc[len(missing_data_df)] = [column_name, missing_data, missing_in_percentage]
    return missing_data_df.sort_values(by = 'percentage_nans')

# ---------------------#
#  Prepare functions   #
# ---------------------#
def create_target_variable_dfs(df):
    '''
    Takes in origional df and creates two dfs that contain only one target variable
    '''
    #Create two dataframes each with only one of the target variables
    h1n1_df = df.drop(columns = 'seasonal_vaccine')

    seasonal_df = df.drop(columns = 'h1n1_vaccine')

    return h1n1_df, seasonal_df

def create_train_test_dfs(h1n1_df, seasonal_df):
    # Use the train test split function from Sklearn and add a random seed for reproducibility
    # Use Stratify y parameter to ensure the same proportion of the y variable in both train and test dfs
    h1n1_train, h1n1_test = train_test_split(h1n1_df, random_state=123, 
    train_size=.80, stratify=h1n1_df.h1n1_vaccine)
    
    seasonal_train, seasonal_test = train_test_split(seasonal_df, random_state=123,
    train_size=.80, stratify=seasonal_df.seasonal_vaccine)

    # Drop these columns as they have too many nans and no good way to fill them
    h1n1_train = h1n1_train.drop(columns =['employment_industry', 'employment_occupation'] )
    h1n1_test = h1n1_test.drop(columns =['employment_industry', 'employment_occupation'] )

    seasonal_train = seasonal_train.drop(columns =['employment_industry', 'employment_occupation'] )
    seasonal_test = seasonal_test.drop(columns =['employment_industry', 'employment_occupation'] )

    return h1n1_train, h1n1_test, seasonal_train, seasonal_test

def fill_null_values(h1n1_train, h1n1_test, seasonal_train, seasonal_test):
    '''
    Fills in null values in train and test dfs with most common value
    '''
    h1n1_train = h1n1_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
    h1n1_test = h1n1_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

    seasonal_train = seasonal_train.apply(lambda x:x.fillna(x.value_counts().index[0]))
    seasonal_test = seasonal_test.apply(lambda x:x.fillna(x.value_counts().index[0]))

    return h1n1_train, h1n1_test, seasonal_train, seasonal_test



# ---------------------#
#    Label Encoding    #
# ---------------------#

def label_encode_columns(h1n1_train, h1n1_test, seasonal_train, seasonal_test):

    encoder = LabelEncoder()
   
    h1n1_train['encoded_employment_status'] = encoder.fit_transform(h1n1_train['employment_status'])
    h1n1_train['encoded_rent_or_own'] = encoder.fit_transform(h1n1_train['rent_or_own'])
    h1n1_train['encoded_marital_status'] = encoder.fit_transform(h1n1_train['marital_status'])
    h1n1_train['encoded_sex'] = encoder.fit_transform(h1n1_train['sex'])

    h1n1_test['encoded_employment_status'] = encoder.fit_transform(h1n1_test['employment_status'])
    h1n1_test['encoded_rent_or_own'] = encoder.fit_transform(h1n1_test['rent_or_own'])
    h1n1_test['encoded_marital_status'] = encoder.fit_transform(h1n1_test['marital_status'])
    h1n1_test['encoded_sex'] = encoder.fit_transform(h1n1_test['sex'])

    seasonal_train['encoded_rent_or_own'] = encoder.fit_transform(seasonal_train['rent_or_own'])
    seasonal_train['encoded_marital_status'] = encoder.fit_transform(seasonal_train['marital_status'])
    seasonal_train['encoded_sex'] = encoder.fit_transform(seasonal_train['sex'])

    seasonal_test['encoded_rent_or_own'] = encoder.fit_transform(seasonal_test['rent_or_own'])
    seasonal_test['encoded_marital_status'] = encoder.fit_transform(seasonal_test['marital_status'])
    seasonal_test['encoded_sex'] = encoder.fit_transform(seasonal_test['sex'])
    
    return h1n1_train, h1n1_test, seasonal_train, seasonal_test

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

def ohe_columns(train, test):
    train, test = ohe_age_group(train, test)
    train, test = ohe_education(train, test)
    train, test = ohe_income_poverty(train, test)
    train, test = ohe_race(train, test)

    return train, test

# ---------------------#
#        Scaling       #
# ---------------------#

def minmax_scale(train, test, column_list):
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


def prepare_data(df, column_list):
    missing_data = percent_nans(df)
    h1n1_df, seasonal_df = create_target_variable_dfs(df)
    h1n1_train, h1n1_test, seasonal_train, seasonal_test = create_train_test_dfs(h1n1_df, seasonal_df)
    h1n1_train, h1n1_test, seasonal_train, seasonal_test = fill_null_values(h1n1_train, h1n1_test, seasonal_train, seasonal_test)
    h1n1_train, h1n1_test, seasonal_train, seasonal_test = label_encode_columns(h1n1_train, h1n1_test, seasonal_train, seasonal_test)
    h1n1_train, h1n1_test = ohe_columns(h1n1_train, h1n1_test)
    seasonal_train, seasonal_test = ohe_columns(seasonal_train, seasonal_test)
    h1n1_train, h1n1_test = minmax_scale(h1n1_train, h1n1_test, column_list)
    seasonal_train, seasonal_test = minmax_scale(seasonal_train, seasonal_test, column_list)

    return h1n1_train, h1n1_test, seasonal_train, seasonal_test
