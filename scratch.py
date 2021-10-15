#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 09:39:03 2021

@author: carolyndavis
"""


# =============================================================================
# IMPORTS 
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as t
from dateutil import parser
#----------------------------------------------------------------------------
# =============================================================================
# Acquiring the Data
# =============================================================================
#Dowloaded data into CSV from Kaggle and saved to proj directory
df = pd.read_csv('atlantic.csv')


#After a Quick Scan of the data:
#Filtering down to only look at data for storms since 1900
collector =[]

for date in df['Date']:
    if date > 19000000:
        collector.append(True)
    else: 
        collector.append(False)
date_range = pd.Series(collector)
new_df = df[date_range]


#-----------------------------------------------------------------------------------
#Another quick look at the data
new_df.shape
#filtered by date 1900-2015



new_df.describe().T

new_df.isnull().sum()
#data is clean and possesses no null values 


# =============================================================================
#           Initial Observations 
# =============================================================================

#Looking at the data:
    #need to remove cols that add no value (Event, id, minimum pressure,  )
    #Atlantic Hurricane** database uses '-999' as a null value. These values must be removed.
    #remove spaces in column names and make lower case
    #change data format for lat and long columns 
    #negative latitude represent the southern hemisphere
    #caribbean coordinates:   21.4691° N, 78.6569° W
    # need to zoom into this area for proj objective 
    #data format needs to be changed 
    #chek for negative values outside of lat and long 
    #lat and long are objects, need data type change after removing cardinals
    #check for any possible duplicate storms in the data 
    
# =============================================================================
# Lil bit of data prep
# =============================================================================

#Removing cardinal letters from the lat and long columns 
new_df = new_df.replace({'N':''}, regex=True)
new_df = new_df.replace({'W':''}, regex=True)
new_df = new_df.replace({'E':''}, regex=True)
new_df = new_df.replace({'S':''}, regex=True)



#drop unnecessary columns 
new_df = new_df.drop(columns=['ID', 'Event', 'Minimum Pressure'], axis =1)

new_df.info()
#-----------------------------------------------------------------------------------
#Renaming the columns to remove spaces and to make lower case:

new_df.columns    

new_df = new_df.rename(columns={'Name': 'name', 'Time': 'time', 'Status': 'status', 'Latitude': 'latitude',
                                'Longitude': 'longitude', 'Maximum Wind': 'max_wind', 'Low Wind NE': 'low_wind_NE',
                                'Low Wind SE': 'low_wind_SE', 'Low Wind SW': 'low_wind_SW', 'Low Wind NW': 'low_wind_NW',
                                'Moderate Wind NE': 'mod_wind_NE', 'Moderate Wind SE': 'mod_wind_SE',
                                'Moderate Wind SW': 'mod_wind_SW', 'Moderate Wind NW': 'mod_wind_NW',
                                'High Wind NE': 'high_wind_NE', 'High Wind SE': 'high_wind_SE', 'High Wind SW': 'high_wind_SW',
                                'High Wind NW': 'high_wind_NW'})

#-----------------------------------------------------------------------------------
new_df.info()


#Convert the lattitude and longitude columns from objects to floats:
#converting Latitude and Longitude to floats:
new_df['latitude'] = new_df['latitude'].astype(float)
new_df['longitude'] = new_df['longitude'].astype(float)

new_df.info()

#-----------------------------------------------------------------------------

#filtering outside latitudes to get desired Caribbean location for hurricane data
lat_filtered = new_df[(new_df['latitude'].astype(float) >= float(9)) & (new_df['latitude'].astype(float) <= float(26))]
lat_filtered.shape

#filtering out longitude by coordinate
#convert longitude to negative
lat_filtered['longitude'] = (lat_filtered['longitude'] * -1)
lat_long_filtered = lat_filtered[(lat_filtered['longitude'] >= -86) & (lat_filtered['longitude'] <= float(-56))]
print(lat_long_filtered.shape)

#-----------------------------------------------------------------------------------
# #Atlantic Hurricane** database uses '-999' as a null value. These values must be removed.

#checking for negative values
negatives =[]
for i in lat_long_filtered['max_wind']:
    if float(i) < 0:
        negatives.append(i)
        
print(negatives)

lat_long_filtered.info()




weather_nulls = ['max_wind', 'low_wind_NE', 'low_wind_SE',
        'low_wind_SW', 'low_wind_NW', 'mod_wind_NE', 'mod_wind_SE', 'mod_wind_SW',
        'mod_wind_NW', 'high_wind_NE', 'high_wind_SE', 'high_wind_SW', 'high_wind_NW']


#Converting all data to strings so can be searched
 # Creates list of all column headers
lat_long_filtered[weather_nulls] = lat_long_filtered[weather_nulls].astype(str)

lat_long_filtered.info()   #all strings good to go 


#Converting all '-999' null values to 'NaN' which Python can automatically remove:
lat_long_filtered = lat_long_filtered.replace('-999', np.nan)

#the year 1967 has '-99' as maximum wind speed values - these must also be changed...dirty data
lat_long_filtered= lat_long_filtered.replace('-99', np.nan)

#double checking it worked like it should:
lat_long_filtered.isnull().sum()
#take away: there were a lot of null values, thank goodness for domain knowledge
            #going to leave them as is for now, may impute, in exploration

#double checking again that there are no negative values in this column:
negatives =[]
for i in lat_long_filtered['max_wind']:
    if float(i) < 0:
        negatives.append(i)
        
print(negatives)
#no negatives looks good 

#-----------------------------------------------------------------------------------
#Changing the Data Format, for future time series exploration:
#first need to convert the object columns back to floats... oops 

# lat_long_filtered.info()

# cols = ['Date','time', 'latitude', 'longitude', 'max_wind', 'low_wind_NE', 'low_wind_SE',
#         'low_wind_SW', 'low_wind_NW', 'mod_wind_NE', 'mod_wind_SE', 'mod_wind_SW',
#         'mod_wind_NW', 'high_wind_NE', 'high_wind_SE', 'high_wind_SW', 'high_wind_NW']

# lat_long_filtered[cols] = lat_long_filtered[cols].astype(float)
#good to move forward


# lat_long_filtered['date'] = pd.to_datetime(lat_long_filtered['Date'].astype(float), format = '%Y %m %d')
lat_long_filtered.info()
#adding year column for further feature analysis

# lat_long_filtered['Year'] = lat_long_filtered['Date'].map(lambda x: x.year)
lat_long_filtered.info()

# lat_long_filtered['date'] = pd.to_datetime(lat_long_filtered['Date']).dt.strftime("%m-%d-%Y")

# lat_long_filtered['year'] = pd.to_datetime(lat_long_filtered['Date']).dt.strftime('%Y')

# 
# lat_long_filtered['month'] = pd.to_datetime(lat_long_filtered['Date']).dt.strftime('%m')

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
        
# lat_long_filtered['Date'] = 
lat_long_filtered = date_fix(lat_long_filtered)
 

lat_long_filtered.info() 
 
 
lat_long_filtered.isnull().sum()
#Based off the data there are just too many null values in the low wind, modwind, and high wind cols
#gonna go ahead and drop these. max wind is my interest as well the status, month, and year cols for
#this project analysis


# =============================================================================
# A Little MoreData Tidy
# =============================================================================
#decided to drop time as well due to my relevancy
#drop unnecessary columns 
clean_df  = lat_long_filtered.drop(columns=['low_wind_NE', 'low_wind_SE',
        'low_wind_SW', 'low_wind_NW', 'mod_wind_NE', 'mod_wind_SE', 'mod_wind_SW',
        'mod_wind_NW', 'high_wind_NE', 'high_wind_SE', 'high_wind_SW', 'high_wind_NW', 'time'], axis =1) 
 



# =============================================================================
# Settomg the index to the datetime for exploration
# =============================================================================

clean_df = clean_df.set_index('Date')




#-----------------------------------------------------------------------------------



# test.info()
# #double checking it worked like it should:
# test.isnull().sum()   #there are still null s but only 41 impute


clean_df['max_wind'] = clean_df['max_wind'].astype(float) 
clean_df = clean_df.sort_values(by='max_wind', ascending=False)



# test = lat_long_filtered.drop_duplicates(subset='name', keep="first")

# #resort by year
# lat_long_filtered = lat_long_filtered.sort_values(by ='year', ascending = True)
# lat_long_filtered.shape

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

clean_df = name_fixer(clean_df, 'name')
clean_df = name_fixer(clean_df, 'status')


def whitespace(data):
    '''
    This function takes in a col from a df and removes 
    any rows with whitespace, maintain index integrity
    '''
    data = data[data['status'].str.len() > 0]
    return data
clean_df = whitespace(clean_df)
# =============================================================================
# More intial thoughts/plans, fixes:
#------------------------------------
    #make cat columns for status of hurrican
    #noticed some whitespace in the status column gonna remove that since its so few
    #interested in using some groupby viz for year 
    #Has there been a steady increase in hurricanes in the last 25 years?
    #there are 5 cats for storms, maybe some new feats based off wind speed?
    
    
    
    
# =============================================================================
# Cat 1 74-95
# Cat 2  96-110
# Cat 3 111-129
# Cat 4 130-156
# Cat 5 > 156    
# =============================================================================
# =============================================================================

# =============================================================================
# Creating some more features for future exploration
# =============================================================================
storms = clean_df['status'].value_counts()

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

clean_df = get_storms(clean_df)

#thats enough features for now. Data is prepped for visualization:
    
#Find the average number of storms over the course of 115 years (1900-2015)
print(9946/115)   
#86.48695652173913 avg number of storms each for this period

print(clean_df['max_wind'].sum()/9946)
# 54.081540317715664 is the avg max wind speed for this period 
    
   
clean_df = clean_df.sort_index()

types = clean_df["status"].unique()

# number of hurricanes
hurricanes = clean_df[clean_df["status"] == "HU"]
hurricanes.loc[:, "ones"] = 1
hurricanes.groupby("year")["ones"].sum().plot(label="num_hurricanes")
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()
plt.close()

def storm_freq_analysis(data, storm):
    storms = data[data["status"] == storm]
    storms.loc[:, "ones"] = 1
    storms.groupby("year")["ones"].sum().plot(label=f"num_of_{storm}")
    plt.legend(bbox_to_anchor=(1.0, 0.5))
    plt.show()
    plt.close()
    
    return storms

storm_freq_collector = {}
for storm in types:
    storm_freq_collector[storm] = storm_freq_analysis(clean_df, str(storm))


# intensity of hurricanes
hurricanes.groupby('year')['max_wind'].median().plot(label='median_max_wind')
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()
plt.close()
    
#how does average number of storms change over the years?
clean_df.groupby('year')['max_wind'].mean().plot(label='max_wind')
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()
plt.close()

#how does average number of storms change over over months?
clean_df.groupby('month')['max_wind'].mean().plot(label='max_wind')
plt.legend(bbox_to_anchor=(1.0, 0.5))
plt.show()
plt.close()
#Takeaway: Can you guess when hurricane/tropical storm season is?

hurdat = hurricanes[['max_wind']] 

#50% of the hurricane data goes to train
train_size = int(len(hurdat)*0.5)



#30% of the hurricane data goes to validate
validate_size = int(len(hurdat)*0.3)



#20% of the hurricane data goes to test
test_size = int(len(hurdat)-train_size - validate_size)


#establishing that the end of the validate set is the length of train size and validate size combined
validate_end_index = train_size + validate_size


#train
train = hurdat[:train_size]
#validate
validate = hurdat[train_size:validate_end_index]
#test
test = hurdat[validate_end_index:]



# is len of train + validate + test == lenght of entire dataframe. 
print(len(train) + len(validate) + len(test) == len(hurdat))

from datetime import timedelta, datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from dateutil import parser
# for tsa 
import statsmodels.api as sm

train2 = train.copy()
train2.index = pd.DatetimeIndex(train.index)

validate2 = validate.copy()
validate2.index = pd.DatetimeIndex(validate.index)


test2 = test.copy()
test2.index = pd.DatetimeIndex(test.index)



#Visualizing the split data:
for col in train2.columns:
    plt.figure(figsize=(12,4))
    plt.plot(train2[col])
    plt.plot(validate2[col])
    plt.plot(test2[col])
    
    plt.ylabel(col)
    plt.title(col)
    
    
    

train2 = pd.DataFrame(train2["max_wind"].resample("Y").median().dropna())


#seasonal decomposition - 1 year
seasonal = sm.tsa.seasonal_decompose(train2["max_wind"], model="additive", period=1)
seasonal.plot()



predict_df = hurricanes[['max_wind']].copy()

train_predict = predict_df[:pd.to_datetime("2014-01-01").date()]
validate_predict = predict_df[pd.to_datetime("2014-01-01").date():pd.to_datetime("2015-01-01").date()]
test_predict = predict_df[pd.to_datetime("2015-01-01").date():pd.to_datetime("2016-01-01").date()]



from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

wind_data = clean_df[['max_wind']].copy()


# def train_validate_test(df, target):
#     '''
#     this function takes in a dataframe and splits it into 3 samples,
#     a test, which is 20% of the entire dataframe,
#     a validate, which is 24% of the entire dataframe,
#     and a train, which is 56% of the entire dataframe.
#     It then splits each of the 3 samples into a dataframe with independent variables
#     and a series with the dependent, or target variable.
#     The function returns train, validate, test sets and also another 3 dataframes and 3 series:
#     X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
#     '''
#     # split df into test (20%) and train_validate (80%)
#     train_validate, test = (train_test_split(df, test_size=.2, random_state=123))
   
#     # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
#     train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    
#     # split train into X (dataframe, drop target) & y (series, keep target only)
#     X_train = train.drop(columns=[target])
#     y_train = train[target]
    
#     # split validate into X (dataframe, drop target) & y (series, keep target only)
#     X_validate = validate.drop(columns=[target])
#     y_validate = validate[target]
    
#     # split test into X (dataframe, drop target) & y (series, keep target only)
#     X_test = test.drop(columns=[target])
#     y_test = test[target]
    
#     return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test



# train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(wind_data, target='max_wind')

wind_data = wind_data.reset_index()
wind_data = wind_data.rename(columns={'Date': 'ds', 'max_wind': 'y'})

from fbprophet import Prophet
model = Prophet(daily_seasonality=True)

model.fit(wind_data)



future = model.make_future_dataframe(periods=365)

prediction = model.predict(future)

model.plot(prediction)



model.plot_components(prediction)
plt.show()


from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
prediction = pd.concat([prediction, wind_data['y']], axis=1)
prediction = prediction.dropna()

r2 = r2_score(prediction['y'], prediction['yhat'])

import math
mse = mean_squared_error(prediction['y'], prediction['yhat'])
rmse = math.sqrt(mse)
