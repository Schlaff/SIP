# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 13:56:36 2021

@author: brian
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 12:49:23 2021

@author: brian
"""

#import Packages
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate


#import NDVI data
NDVI = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Sandy_3_Allotment\S3_WB_NDVI.csv')

#EVI data to build same relationships
#NDVI = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Sandy_3_Allotment\S3_WB_EVI.csv')
#NDVI['scaled NDVI3'] = NDVI['scaled EVI3']

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


#Create a blank dataframe that stores the best relationship slope intercept and r2 value
Best = pd.DataFrame()
Best['Date Range'] = Dates
Best['Period'] = Period
Best['Best Predictor(s)'] = ''
Best['r^2'] = np.nan
Best['rMSE'] = np.nan

    

# loop through the data and Calculate R2 and MSE for all of the different periods

for i in range(5,20,1):
    
    NDVI_p = NDVI[NDVI["Period"] == i]
    
    # Create a liear regression model for all the predictor tests
    linreg = LinearRegression()
    
    ### Deficit Vs. NDVI
    #Split the data
    linregD = LinearRegression()
    XD = pd.DataFrame(NDVI_p['Sum of D'])
    yD = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_D = cross_validate(linregD, XD, yD, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2D = np.mean(scores_D['train_r2'])
    rMSED = abs(np.mean(scores_D['test_neg_mean_squared_error']))
    
    ### AET Vs. NDVI
    #Split the data
    linregA = LinearRegression()
    XA = pd.DataFrame(NDVI_p['Sum of AET'])
    yA = pd.DataFrame(NDVI_p['scaled NDVI3'])
    linA = linreg.fit(XA, yA)
    
    #Run K-fold Cross Validation (K=5)
    scores_A = cross_validate(linregA, XA, yA, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2A = np.mean(scores_A['train_r2'])
    rMSEA = abs(np.mean(scores_A['test_neg_mean_squared_error']))
    
    ### Soil Moisture Vs. NDVI
    #Split the data
    linregS = LinearRegression()
    XS = pd.DataFrame(NDVI_p['Average of SOIL'])
    yS = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_S = cross_validate(linregS, XS, yS, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2S = np.mean(scores_S['train_r2'])
    rMSES = abs(np.mean(scores_S['test_neg_mean_squared_error']))
    
    ### Melt Vs. NDVI
    #Split the data
    linregM = LinearRegression()
    XM = pd.DataFrame(NDVI_p['Sum of MELT'])
    yM = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_M = cross_validate(linregM, XM, yM, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2M = np.mean(scores_M['train_r2'])
    rMSEM = abs(np.mean(scores_M['test_neg_mean_squared_error']))
    
    ### W Vs. NDVI
    #Split the data
    linregW = LinearRegression()
    XW = pd.DataFrame(NDVI_p['Sum of W'])
    yW = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_W = cross_validate(linregW, XW, yW, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2W = np.mean(scores_W['train_r2'])
    rMSEW = abs(np.mean(scores_W['test_neg_mean_squared_error']))

    
    ### Precipitation Vs. NDVI
    #Split the data
    linregP = LinearRegression()
    XP = pd.DataFrame(NDVI_p['Sum of P'])
    yP = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_P = cross_validate(linregP, XP, yP, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate r2 and rMSE values
    r2P = np.mean(scores_P['train_r2'])
    rMSEP = abs(np.mean(scores_P['test_neg_mean_squared_error']))
    
    ### Temperature Precipitation Vs. NDVI
    #Split the data
    linregTP = LinearRegression()
    XTP = pd.DataFrame(NDVI_p[['Average of T','Sum of P']])
    yTP = pd.DataFrame(NDVI_p['scaled NDVI3'])
    
    #Run K-fold Cross Validation (K=5)
    scores_TP = cross_validate(linregTP, XTP, yTP, cv=5,
                        scoring=('r2', 'neg_mean_squared_error'),
                        return_train_score=True)
    
    #calculate adjusted r2 and rMSE values
    n = len(NDVI_p)
    p = 2
    r2TP = np.mean(scores_TP['train_r2'])
    adj_r2TP = 1-((1-r2TP)*(n-1)/(n-p-1))
    rMSETP = abs(np.mean(scores_TP['test_neg_mean_squared_error']))

    ###Figure out the best model fit for each period
    list = [r2D, r2A, r2S, r2M, r2W, r2P, adj_r2TP]
    best_r2 = list.index(max(list))
    
    if best_r2 == 0:
        linregD.fit(XD,yD)
        Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
        Best.at[i-5,'r^2'] = r2D
        Best.at[i-5,'rMSE'] = rMSED
        Best.at[i-5,'Intercept'] = float(linregD.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregD.coef_[0])
         
    elif best_r2 ==1:
        linregA.fit(XA,yA)
        Best.at[i-5,'Best Predictor(s)'] = 'AET'
        Best.at[i-5,'r^2'] = r2A
        Best.at[i-5,'rMSE'] = rMSEA
        Best.at[i-5,'Intercept'] = float(linregA.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregA.coef_[0])
           
    elif best_r2 ==2:
        linregS.fit(XS,yS)
        Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
        Best.at[i-5,'r^2'] = r2S
        Best.at[i-5,'rMSE'] = rMSES
        Best.at[i-5,'Intercept'] = float(linregS.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregS.coef_[0])
        
    elif best_r2 ==3:
        linregM.fit(XM,yM)
        Best.at[i-5,'Best Predictor(s)'] = 'Snow Melt (M)'
        Best.at[i-5,'r^2'] = r2M
        Best.at[i-5,'rMSE'] = rMSEM
        Best.at[i-5,'Intercept'] = float(linregM.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregM.coef_[0])
            
    elif best_r2 ==4:
        linregW.fit(XW,yW)
        Best.at[i-5,'Best Predictor(s)'] = 'W'
        Best.at[i-5,'r^2'] = r2W
        Best.at[i-5,'rMSE'] = rMSEW
        Best.at[i-5,'Intercept'] = float(linregW.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregW.coef_[0])
        
    elif best_r2 ==5:
        linregP.fit(XP,yP)
        Best.at[i-5,'Best Predictor(s)'] = 'Precip'
        Best.at[i-5,'r^2'] = r2P
        Best.at[i-5,'rMSE'] = rMSEP
        Best.at[i-5,'Intercept'] = float(linregP.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregP.coef_[0])
        
    elif best_r2 ==6:
        linregTP.fit(XTP,yTP)
        Best.at[i-5,'Best Predictor(s)'] = 'Temperature + Precipitation'
        Best.at[i-5,'r^2'] = adj_r2TP
        Best.at[i-5,'rMSE'] = rMSETP
        Best.at[i-5,'Intercept'] = float(linregTP.intercept_[0])
        Best.at[i-5,'Coefficient'] = float(linregTP.coef_[0])
        Best.at[i-5,'Coefficient 2'] = float(linregTP.coef_[1])
