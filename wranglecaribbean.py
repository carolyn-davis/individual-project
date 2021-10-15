#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 06:25:15 2021

@author: carolyndavis
"""

# =============================================================================
# #FUNCTIONS FOR CARIBBEAN STORM PROJECT
# =============================================================================
#Imports:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as t
from dateutil import parser
import warnings
warnings.filterwarnings("ignore")

# visualize 
# %matplotlib inline
import seaborn as sns


# working with dates
from datetime import datetime

# to evaluated performance using rmse
from sklearn.metrics import mean_squared_error
from math import sqrt 

# for tsa 
import statsmodels.api as sm

# for fbprophet modeling
from fbprophet import Prophet
from sklearn.metrics import r2_score
import math




# =============================================================================
# FUNCTIONS FOR DATA PREP 
# =============================================================================
def date_fix(data):
    ''' This function takes in the date column and converts it to datetime format
       It additionally adds momth and columns by splitting the converted date column
       into month years based on the original format of the date col
    '''
    
    
    data = data.reset_index(drop=True)
    collector = []
    for date in data['Date']:
        date = parser.parse(str(date))
        collector.append(date.date())
        
    collector = pd.Series(collector)
    data['Date'] = collector 
    
    def new_cols(data):
        month_collector = []
        year_collector = []
        for date in data['Date']:
            date = str(date)
            year, month, day = date.split('-')
            month_collector.append(month)
            year_collector.append(year)
            
        data['month'] = pd.Series(month_collector)
        data['year'] = pd.Series(year_collector)
        
        return data
            
    data = new_cols(data)
        
    return data





def name_fixer(data, col):
    collector = []
    for name in data[col]:
        name = str(name)
        name = name.strip()
        collector.append(name)
    collector = pd.Series(collector)
    collector.index = data.index
    data[col] = collector
    
    return data


def whitespace(data):
    '''
    This function takes in a col from a df and removes 
    any rows with whitespace, maintain index integrity
    '''
    data = data[data['status'].str.len() > 0]
    return data



def get_storms(storm_df):
    '''
    This function will create dummy variables out of the original status column.
    It takes in the df column and converts the status types to booleans.
    '''
    # create dummy vars of storms id
    storm_df = pd.get_dummies(storm_df.status)
    # rename columns by actual county name
    storm_df.columns = ['T', 'HU', 'TD', 'LO', 'V', 'DB', 'X', 'D']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    storm_df = pd.concat([clean_df, storm_df], axis = 1)
    # drop  status columns
    # df = df.drop(columns = ['status'])
    return storm_df



def storm_freq_analysis(data, storm):
    storms = data[data["status"] == storm]
    storms.loc[:, "ones"] = 1
    storms.groupby("year")["ones"].sum().plot(label=f"num_of_{storm}")
    plt.legend(bbox_to_anchor=(1.0, 0.5))
    plt.show()
    plt.close()
    
    return storms

# =============================================================================
# Function for Exploration
# =============================================================================
def correlation(y, lag):
    return pd.concat([y, y.shift(lag)], axis=1).dropna().corr().iloc[0, 1]




# =============================================================================
# Functions for Modeling Stage
# =============================================================================
def evaluate(target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate_hurdat[target_var], yhat_df[target_var])), 0)
    return rmse


def plot_and_eval(target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train_hurdat[target_var], label='Train', linewidth=1)
    plt.plot(validate_hurdat[target_var], label='Validate', linewidth=1)
    plt.plot(yhat_df[target_var])
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()
    
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def make_predictions():
    '''
    make a prediction column
    '''
    yhat_df = pd.DataFrame({'max_wind': [temp]},  
                      index = validate_hurdat.index)

    return yhat_df

def final_plot(target_var):
    plt.figure(figsize=(12,4))
    plt.plot(train_hurdat[target_var], label='train')
    plt.plot(validate_hurdat[target_var], label='validate')
    plt.plot(test_hurdat[target_var], label='test')
    plt.plot(yhat_df[target_var], alpha=.5)
    plt.title(target_var)
    plt.show()