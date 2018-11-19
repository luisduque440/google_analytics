import datetime as dt
from datetime import timedelta

import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn import preprocessing


def get_training_data(df , first_train_day):
    """ 

    Args:
        df: 
        first_train_day:
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """
    last_train_day = first_train_day + timedelta(days=168)
    first_test_day = last_train_day + timedelta(days=46)
    last_test_day = first_test_day + timedelta(days=62)

    train_visits = df[(df.date>= first_train_day) & (df.date<last_train_day)]
    test_visits = df[(df.date>=first_test_day) & (df.date<last_test_day)]

    y = get_target(train_visits, test_visits)
    X = create_features(train_visits)
    X['target']=y.target
    return X

def create_features(df):
    """ feature engineering... constant fields + average last months + months ago

    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """
    last_train_day = df.date.max()
    feature_dfs=[]

    feature_dfs.append(get_fixed_fields(df)) 

    for i in [1,2,3,6]:
        start = last_train_day - timedelta(days=i*30)
        end = last_train_day
        suffix = "_last_%s_months" %(i)
        feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix=suffix))

    for i in [2,3,4]:
        start = last_train_day - timedelta(days=i*30)
        end = start + timedelta(days=30)
        suffix = "_%s_months_ago" %(i)
        feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix=suffix))
    
    return pd.concat(feature_dfs, axis=1)



def get_cummulate_numeric_fields(df, start_date, end_date, suffix=''):
    """ cummulate of numeric fields 

    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df and
            one column per float64 field
    """
    numeric_column_names = df.select_dtypes(include='float64').columns
    
    result = pd.DataFrame(index=df.fullVisitorId.unique())
    result.index.name= 'fullVisitorId'

    relevant_df = df[(df.date>= start_date)& (df.date<=end_date)]
    different_visitors = relevant_df.groupby('fullVisitorId')
    
    for col in numeric_column_names:
        cummulative_sum = different_visitors[col].sum()
        result[col+suffix] = cummulative_sum
    return result.fillna(0)



def get_fixed_fields(df, suffix=''):
    """ mode of all  
 
    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df and
            one column per object field
    """
    
    object_column_names = list(df.select_dtypes(include='object').columns) 
    object_column_names.remove('fullVisitorId')
    
    result = pd.DataFrame(index=df.fullVisitorId.unique())
    result.index.name= 'fullVisitorId'
    different_visitors = df.groupby('fullVisitorId')
    
    for col in object_column_names:
        most_common = different_visitors[col].last()
        result[col+suffix] = most_common
    return result


def get_target(train_visits, test_visits):
    """gets the target.
 
    Args:
        train_visits: DataFrame. Each row is a visit in the past
        test_visits: DataFrame. Each row is a visit in the future

 
    Returns:
        DataFrame: one row for each fullVisitorId in train_visit and
            one column (log_total_spent) with the natural log 
            of the sum of all transactions in test_visits dataframe
    """
    target = pd.DataFrame(index=train_visits.fullVisitorId.unique())
    target.index.name = 'fullVisitorId'
    total_spent = test_visits.groupby('fullVisitorId')['totals_totalTransactionRevenue'].agg(np.sum)
    target['total_spent']=total_spent
    target = target.fillna(0)
    target['target']= target.total_spent.apply(lambda x: np.log(x+1))
    target=target.drop(columns=['total_spent'])
    return target



def label_encode_object_dtypes(df):
    """Label encodes all the columns of a DataFrames df1 and df2 that have
       dtype='object'
 
    Args:
        df
        
    Returns:
        DataFrame: df with object types encoded.
    """
    object_column_names = list(df.select_dtypes(include='object').columns)    
    
    for col in object_column_names:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col].values)
        df[col] = lbl.transform(df[col].values)
    return df



def fill_empty_values(df):
    """'
    Args:
        df1
        
    Returns:
        DataFrame: df1 with filled empty values and columns types formated
    """
    df['totals_newVisits'] = df['totals_newVisits'].astype('object')
    df['totals_bounces'] = df['totals_bounces'].astype('object')
    df['device_isMobile'] = df['device_isMobile'].astype('object')

    object_column_names = df.select_dtypes(include='object').columns 
    for col in object_column_names:
        df[col] = df[col].apply(unicode).apply(lambda x:x.encode('utf-8'))

    numeric_column_names = set(df.columns).difference(set(object_column_names))
    for col in numeric_column_names:
        df[col] = df[col].astype('float64')

    df['date'] = pd.to_datetime(df.visitStartTime, unit='s').astype('datetime64')
    df = df.drop(columns = ['visitStartTime', 'visitId'])
    df = df.fillna(0)
    return df



def get_basic_info(df):
    """gets basic information from a data frame
 
    Args:
        df
        
    Returns:
        A dataframe that has one row for each column in df1 and 5 columns:
        unique_elements(the number of unique values), 'mode' (most common element), 
        'empty_values' (number of empty values), 'dtype', 'types' (python types
        contained in that column)
    """
    colnames = df.columns
    Info= pd.DataFrame({
        'column_name': colnames, 
        'unique_elements': [df[col].nunique() for col in colnames],
        'mode': [df[col].mode()[0] for col in colnames],
        'empty_values': [df[col].isna().sum() for col in colnames],
        'dtype': df.dtypes.values,
        'types': [df[col].apply(type).value_counts().to_dict() for col in colnames]
    })
    return Info.set_index('column_name')

