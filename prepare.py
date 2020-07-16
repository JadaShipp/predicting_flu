import pandas as import pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



def label_encode_columns(df):
    train, test = train_test_split(df, random_state=123, train_size=.80)

    encoder = LabelEncoder()

    encode_list = [
    'gender', 'partner', 'dependents', 'phone_service'
    , 'multiple_lines', 'online_security', 'online_backup'
    , 'device_protection', 'tech_support'
    , 'streaming_movies', 'streaming_tv', 'paperless_billing', 'churn'
    ]
             
    for e in encode_list:
        train[e] = encoder.fit_transform(train[e])
        test[e] = encoder.transform(test[e])

def one_hot_encode_columns(df):
    encoder = OneHotEncoder()

    encode_list = [
        ''
    ]