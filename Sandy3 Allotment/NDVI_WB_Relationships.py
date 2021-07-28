# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:49:23 2021

@author: brian
"""

#import Packages
import csv
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


#import EVI data
NDVI = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Sandy_3_Allotment\S3_WB_NDVI.csv')

#change date to datetime
NDVI['date'] = pd.to_datetime(NDVI['date']).dt.tz_localize(None)

#drop NA values
NDVI.dropna(subset = ['scaled NDVI3'], inplace = True)


###Calculate R2 and MSE for periodic values

#create blank dataframe of just growing season data
GS = pd.DataFrame()
Period = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
Dates = ['March 6 - March 21', 'March 22 - April 6', 'April 7 - April 22', 'April 23 - May 8', 'May 9 - May 24', 'May 25 - June 9', 'June 10 - June 25', 'June 26 - July 11', 'July 12 - July 27', 'July 28 - August 12', 'August 13 - August 28', 'August 29 - September 13', 'September 14 - September 29', 'September 30 - October 14', 'October 15 - October 31']
GS['Date Range'] = Dates
GS['Period'] = Period
GS['r^2 of Deficit vs. EVI'] = np.nan
GS['r^2 of AET vs. EVI'] = np.nan
GS['r^2 of SM vs. EVI'] = np.nan
GS['r^2 of MELT vs. EVI'] = np.nan
GS['r^2 of W vs. EVI'] = np.nan
GS['rMSE of Deficit vs. EVI'] = np.nan 
GS['rMSE of AET vs. EVI'] = np.nan
GS['rMSE of SM vs. EVI'] = np.nan
GS['rMSE of MELT vs. EVI'] = np.nan
GS['rMSE of W vs. EVI'] = np.nan
GS['AIC of Deficit vs. EVI'] = np.nan 
GS['AIC of AET vs. EVI'] = np.nan
GS['AIC of SM vs. EVI'] = np.nan
GS['AIC of MELT vs. EVI'] = np.nan
GS['AIC of W vs. EVI'] = np.nan

#Create a blank dataframe that stores the best relationship slope intercept and r2 value
Best = pd.DataFrame()
Best['Date Range'] = Dates
Best['Period'] = Period
Best['Best Predictor(s)'] = ''
Best['r^2'] = np.nan
Best['rMSE'] = np.nan
Best['AIC'] = np.nan
    

# loop through the data and Calculate R2 and MSE for all of the different periods

for i in range(5,20,1):
    
    EVI_p = NDVI[NDVI["Period"] == i]
    
    
    ####Deficit Vs. EVI
    X_trainD, X_testD, y_trainD, y_testD = train_test_split(EVI_p['Sum of D'], EVI_p['scaled NDVI3'], test_size=0.25, random_state=12)

    #Establish a linear relationship between test and training data
    slopeD, interceptD, r_valueD, p_valueD, std_errD = scipy.stats.linregress(X_trainD, y_trainD)
    r2D = ((r_valueD**2))
    GS.at[i-5,'r^2 of Deficit vs. EVI'] =r2D

    #try relationship with test data
    y_predictD = (slopeD*X_testD) + interceptD

    #calculate MSE between y_predict and y_test
    AICD = 2 - (2*np.log((sum((y_testD-y_predictD)**2))))
    GS.at[i-5,'AIC of Deficit vs. EVI'] = AICD
    rMSED = sqrt(mean_squared_error(y_testD, y_predictD))
    GS.at[i-5,'rMSE of Deficit vs. EVI'] = rMSED


    ###AET vs EVI
    X_trainA, X_testA, y_trainA, y_testA = train_test_split(EVI_p['Sum of AET'], EVI_p['scaled NDVI3'], test_size=0.25, random_state=12)

    #Establish a linear relationship between test and training data
    slopeA, interceptA, r_valueA, p_valueA, std_errA = scipy.stats.linregress(X_trainA, y_trainA)
    r2A = ((r_valueA**2))
    GS.at[i-5,'r^2 of AET vs. EVI'] =r2A

    #try relationship with test data
    y_predictA = (slopeA*X_testA) + interceptA

    #calculate MSE between y_predict and y_test
    AICA = 2 - (2*np.log((sum((y_testA-y_predictA)**2))))
    GS.at[i-5,'AIC of AET vs. EVI'] = AICA
    rMSEA = sqrt(mean_squared_error(y_testA, y_predictA))
    GS.at[i-5,'rMSE of AET vs. EVI'] = rMSEA


    ###SM vs. EVI
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(EVI_p['Average of SOIL'], EVI_p['scaled NDVI3'], test_size=0.25, random_state=12)

    #Establish a linear relationship between test and training data
    slopeS, interceptS, r_valueS, p_valueS, std_errS = scipy.stats.linregress(X_trainS, y_trainS)
    r2S = ((r_valueS**2))
    GS.at[i-5,'r^2 of SM vs. EVI'] =r2S

    #try relationship with test data
    y_predictS = (slopeS*X_testS) + interceptS

    #calculate MSE between y_predict and y_test
    AICS = 2 - (2*np.log((sum((y_testS-y_predictS)**2))))
    GS.at[i-5,'AIC of SM vs. EVI'] = AICS
    rMSES = sqrt(mean_squared_error(y_testS, y_predictS))
    GS.at[i-5,'rMSE of SM vs. EVI'] = rMSES
    
    
    ###W vs. EVI
    X_trainW, X_testW, y_trainW, y_testW = train_test_split(EVI_p['Sum of W'], EVI_p['scaled NDVI3'], test_size=0.25, random_state=12)

    #Establish a linear relationship between test and training data
    slopeW, interceptW, r_valueW, p_valueW, std_errW = scipy.stats.linregress(X_trainW, y_trainW)
    r2W = ((r_valueW**2))
    GS.at[i-5,'r^2 of W vs. EVI'] =r2W

    #try relationship with test data
    y_predictW = (slopeW*X_testW) + interceptW

    #calculate MSE between y_predict and y_test
    AICW = 2 - (2*np.log((sum((y_testW-y_predictW)**2))))
    GS.at[i-5,'AIC of W vs. EVI'] = AICW
    rMSEW = sqrt(mean_squared_error(y_testW, y_predictW))
    GS.at[i-5,'rMSE of W vs. EVI'] = rMSEW
    
    ###Figure out the best model fit for each period
    list = [r2D, r2A, r2S, r2W]
    best_r2 = list.index(max(list))
    
    if best_r2 == 0:
        Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
        Best.at[i-5,'r^2'] = r2D
        Best.at[i-5,'Intercept'] = interceptD
        Best.at[i-5,'Coefficient'] = slopeD
        Best.at[i-5,'AIC'] = AICD
        Best.at[i-5,'rMSE'] = rMSED
        
    
    elif best_r2 ==1 :
        Best.at[i-5,'Best Predictor(s)'] = 'AET'
        Best.at[i-5,'r^2'] = r2A
        Best.at[i-5,'Intercept'] = interceptA
        Best.at[i-5,'Coefficient'] = slopeA
        Best.at[i-5,'AIC'] = AICA
        Best.at[i-5,'rMSE'] = rMSEA
        
    
    elif best_r2 ==2:
        Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
        Best.at[i-5,'r^2'] = r2S
        Best.at[i-5,'Intercept'] = interceptS
        Best.at[i-5,'Coefficient'] = slopeS
        Best.at[i-5,'AIC'] = AICS
        Best.at[i-5,'rMSE'] = rMSES
        
        
    elif best_r2 ==3:
        Best.at[i-5,'Best Predictor(s)'] = 'W'
        Best.at[i-5,'r^2'] = r2W
        Best.at[i-5,'Intercept'] = interceptW
        Best.at[i-5,'Coefficient'] = slopeW
        Best.at[i-5,'AIC'] = AICW
        Best.at[i-5,'rMSE'] = rMSEW
