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
    storm_df = pd.get_dummies(clean_df.status)
    # rename columns by actual county name
    storm_df.columns = ['T', 'HU', 'TD', 'LO', 'V', 'DB', 'X', 'D']
    # concatenate the dataframe with the 3 county columns to the original dataframe
    storm_df = pd.concat([clean_df, storm_df], axis = 1)
    # drop  status columns
    # df = df.drop(columns = ['status'])
    return df

test = get_storms(clean_df)






#Takeaways:
#left with 226 storms over 65 year period
#What is the average number of storm per year for this period?
#Wind speed is indicative of storms, what is the average wind speed over this period?

#Find the average number of storms over the course of 65 years (1950-2015)
print(226/65)
#3.476923076923077 avg num of storms

#finding the avg wind speed for storms over the course of 65 years:


#-----------------------------------------------------------------------------------
#A few more tweeks for columns 

#gonna drop this col because focusing on month/year storm analysis
lat_long_filtered = lat_long_filtered.drop(columns=['time', 'date'])

#-----------------------------------------------------------------------------------
#looking at the avg wind speed for storms over the last 65 years

print(lat_long_filtered['max_wind'].sum()/226)
# 68.25221238938053 mph


#-----------------------------------------------------------------------------------

#creating dataframe of number of storms per year

#looking at the number of storms for each year 
count = lat_long_filtered['Year'].value_counts()


#change the couunt from a series to a dataframe
count_df = pd.DataFrame(count)

#reset the index so that it not the year
count_df = count_df.reset_index()


#rename the columns for readibility since the index col is now the year
count_df = count_df.rename(columns={"index": "Year", "Year": "Count"})

count_df = count_df.sort_values(by = "Year", ascending = True)
#Takeaways:
    #it appears tht storm count has increased significantly in the last 25 years 

#-----------------------------------------------------------------------------------
#Summmarizing this Data and looking at the averages for storms each year


grouped_df = lat_long_filtered.groupby(['Year'])

described_df = grouped_df.describe()

described_df = described_df.reset_index()


described_df = pd.DataFrame(described_df)

described_df.columns = ['Year', 'Count', "Mean", 'std', 'min', '25%', '50%', '75%', 'Max']

described_df


#-----------------------------------------------------------------------------------
#adding month column to new dataframe
month_df = lat_long_filtered
month_df = month_df.drop(['Year'], axis=1)

month_df['Month'] = month_df['Date'].map(lambda x: x.month)
#############

#group by month

# month_df['Maximum Wind'] = month_df['Maximum Wind'].astype(float)
month_df.Month = month_df.Month.astype(int)

grouped_df_month = month_df.groupby(['Month'])

described_df_month = grouped_df_month.describe()

described_df_month = described_df_month.reset_index()


described_df_month= pd.DataFrame(described_df_month)

described_df_month.columns = ['Month', 'Count', "Mean", 'std', 'min', '25%', '50%', '75%', 'Max']

described_df_month

