import datetime as dt
from datetime import timedelta

import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin  


def get_training_data_from_visits(df):
    """ Notice that 276 days is around 9months (our training+testing period)

    Args:
        df: 
        first_train_day:
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """

    first_train_day = df.date.min()
    last_train_day = df.date.max() + timedelta(days=1) 

    all_training_periods = []
    while(first_train_day + timedelta(days=276) <= last_train_day):  
        training_data = get_training_data_in_a_period(df , first_train_day)
        all_training_periods.append(training_data.reset_index())
        first_train_day = first_train_day + timedelta(days=28)
    return pd.concat(all_training_periods, ignore_index=True)

def get_training_data_in_a_period(df , first_train_day):
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

    train_period = df[(df.date>= first_train_day) & (df.date<last_train_day)]
    test_period = df[(df.date>=first_test_day) & (df.date<last_test_day)]

    y = get_target(train_period, test_period)
    X = create_features(train_period)
    X['target']=y.target
    return X

def create_features(df):
    """ feature engineering... constant fields + average last months + months ago

    Args:
        df: 
        
    Returns:
        DataFrame: one row for each fullVisitorId in df
    """
    last_day = df.date.max() 
    feature_dfs=[]

    feature_dfs.append(get_fixed_fields(df)) 

    for i in [1,3,4,6]:
        start = last_day - timedelta(days=i*28)
        end = last_day
        feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix="_last_%s_months" %(i)))

    for i in [2,3,4]:
        start = last_day - timedelta(days=i*28)
        end = start + timedelta(days=28)
        feature_dfs.append(get_cummulate_numeric_fields(df, start, end, suffix="_%s_months_ago" %(i)))
    
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
    
    grouped_df = relevant_df.groupby('fullVisitorId')
    for col in numeric_column_names:
        result[col+suffix] = grouped_df[col].sum()
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
    
    grouped_df = df.sort_values(by=['date']).groupby('fullVisitorId')
    for col in object_column_names:
        result[col+suffix] = grouped_df[col].last()
    return result


def get_target(train_period, test_period):
    """gets the target.
 
    Args:
        train_period: DataFrame. Each row is a visit in the past
        test_period: DataFrame. Each row is a visit in the future

 
    Returns:
        DataFrame: one row for each fullVisitorId in train_visit and
            one column (log_total_spent) with the natural log 
            of the sum of all transactions in test_period dataframe
    """
    target = pd.DataFrame(index=train_period.fullVisitorId.unique())
    target.index.name = 'fullVisitorId'
    target['total_spent'] = test_period.groupby('fullVisitorId')['totals_transactionRevenue'].agg(np.sum)
    target = target.fillna(0)
    target['target']= target.total_spent.apply(lambda x: np.log(x+1))
    target['month']= str(test_period.date.min().month)

    return target.drop(columns=['total_spent'])



class CategoricalEncoder(TransformerMixin):  
    """Label encodes all the columns of a DataFrames df1 and df2 that have
       dtype='object'
 
    Args:
        df
        
    Returns:
        DataFrame: df with object types encoded.
    """
    def fit(self, X, y=None):
        self.categorical_columns = list(X.select_dtypes(include='object').columns)
        self.categorical_columns.remove('fullVisitorId')
        self.level_dictionary={}
        for col in self.categorical_columns:
            importance = X.groupby(col)['totals_transactionRevenue'].agg([np.sum, np.size])
            self.level_dictionary[col] = 1.0*importance['sum']/(importance['size']+1) 
        return self
    
    def transform(self, X):
        df=X.copy()
        dictionary = self.level_dictionary
        for col in self.categorical_columns:
            df[col] = df[col].apply(lambda x: dictionary[col][x] if x in dictionary[col].index else 0)
        return df


def fill_empty_values(df):
    """'
    Args:
        df1
        
    Returns:
        DataFrame: df1 with filled empty values and columns types formated
    """
    
    object_column_names = df.select_dtypes(include='object').columns 
    for col in object_column_names:
        df[col] = df[col].apply(unicode).apply(lambda x:x.encode('utf-8'))

    numeric_column_names = set(df.columns).difference(set(object_column_names))
    for col in numeric_column_names:
        df[col] = df[col].astype('float64')

    df['date'] = pd.to_datetime(df.visitStartTime, unit='s').astype('datetime64')
    df['date'] = df.date.apply(lambda x: dt.datetime(x.year, x.month, x.day))
    df = df.drop(columns = ['visitStartTime', 'visitId'])
    df = df.fillna(0)
    df['visits'] = 1.0 
    df['paying_visits'] = (df.totals_transactionRevenue>0).apply(float)
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


def get_visits_from_time_period(df, start_date, end_date):
    dg=df.copy()
    return dg[(dg.date >= start_date) & (dg.date<end_date)]

