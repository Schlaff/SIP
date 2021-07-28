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
from sklearn.model_selection import cross_val_score


#import EVI data
EVI = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Sandy_3_Allotment\S3_WB_EVI.csv')

#change date to datetime
EVI['date'] = pd.to_datetime(EVI['date']).dt.tz_localize(None)

#drop NA values
EVI.dropna(subset = ['scaled EVI3'], inplace = True)

#check the correlation of variables
f = plt.figure(figsize=(19, 15))
plt.matshow(EVI.corr(), fignum=f.number)
plt.xticks(range(EVI.select_dtypes(['number']).shape[1]), EVI.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(EVI.select_dtypes(['number']).shape[1]), EVI.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

#split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(EVI['Sum of AET'], EVI['scaled EVI3'], test_size=0.25, random_state=12)

#Establish a linear relationship between test and training data
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(X_train, y_train)
r2 = round((r_value**2),4)
print('r2 of AET to EVI')
print(r2)

#try relationship with test data
y_predict = (slope*X_test) + intercept

#calculate MSE between y_predict and y_test
print('root MSE of EVI and AET relationship')
print(sqrt(mean_squared_error(y_test,y_predict)))


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
GS['Adj. r2 of D + AET + SM vs. EVI'] = np.nan
GS['Adj. r2 of D + AET vs. EVI'] = np.nan
GS['Adj. r2 of AET + SM vs. EVI'] = np.nan
GS['Adj. r2 of D + SM vs. EVI'] = np.nan
GS['rMSE of Deficit vs. EVI'] = np.nan 
GS['rMSE of AET vs. EVI'] = np.nan
GS['rMSE of SM vs. EVI'] = np.nan
GS['rMSE of MELT vs. EVI'] = np.nan
GS['rMSE of D + AET + SM vs. EVI'] = np.nan
GS['rMSE of D + AET vs. EVI'] = np.nan
GS['rMSE of AET + SM vs. EVI'] = np.nan
GS['rMSE of D + SM vs. EVI'] = np.nan
GS['AIC of Deficit vs. EVI'] = np.nan 
GS['AIC of AET vs. EVI'] = np.nan
GS['AIC of SM vs. EVI'] = np.nan
GS['AIC of MELT vs. EVI'] = np.nan
GS['AIC of D + AET + SM vs. EVI'] = np.nan
GS['AIC of D + AET vs. EVI'] = np.nan
GS['AIC of AET + SM vs. EVI'] = np.nan
GS['AIC of D + SM vs. EVI'] = np.nan

#Create a blank dataframe that stores the best relationship slope intercept and r2 value
Best = pd.DataFrame()
Best['Date Range'] = Dates
Best['Period'] = Period
Best['Best Predictor(s)'] = ''
Best['r^2'] = np.nan
Best['adjusted r^2'] = np.nan
Best['rMSE'] = np.nan
Best['AIC'] = np.nan
    

# loop through the data and Calculate R2 and MSE for all of the different periods

for i in range(5,20,1):
    
    EVI_p = EVI[EVI["Period"] == i]
    
    ####Deficit Vs. EVI
    X_trainD, X_testD, y_trainD, y_testD = train_test_split(EVI_p['Sum of D'], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)

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
    X_trainA, X_testA, y_trainA, y_testA = train_test_split(EVI_p['Sum of AET'], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)

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
    X_trainS, X_testS, y_trainS, y_testS = train_test_split(EVI_p['Average of SOIL'], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)

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
    
     ###SM vs. EVI
    X_trainSn, X_testSn, y_trainSn, y_testSn = train_test_split(EVI_p['Sum of MELT'], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)

    #Establish a linear relationship between test and training data
    slopeSn, interceptSn, r_valueSn, p_valueSn, std_errSn = scipy.stats.linregress(X_trainSn, y_trainSn)
    r2Sn = ((r_valueSn**2))
    GS.at[i-5,'r^2 of MELT vs. EVI'] =r2Sn

    #try relationship with test data
    y_predictSn = (slopeSn*X_testSn) + interceptSn

    #calculate MSE between y_predict and y_test
    AICSn = 2 - (2*np.log((sum((y_testSn-y_predictSn)**2))))
    GS.at[i-5,'AIC of MELT vs. EVI'] = AICSn
    rMSESn = sqrt(mean_squared_error(y_testSn, y_predictSn))
    GS.at[i-5,'rMSE of MELT vs. EVI'] = rMSESn
    
    ### D + AET + SM vs. EVI
    X_trainm, X_testm, y_trainm, y_testm = train_test_split(EVI_p[['Sum of D','Sum of AET','Average of SOIL']], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)
    
    #estrablish multivariable linear relationship 
    model = LinearRegression()
    model.fit(X_trainm, y_trainm)
    
    #calculated adjusted r2 value
    yhat = model.predict(X_trainm)
    SS_Residual = sum((y_trainm-yhat)**2)       
    SS_Total = sum((y_trainm-np.mean(y_trainm))**2)     
    r_squared = 1 - (float(SS_Residual))/SS_Total
    adj_r2 = 1 - (((1-r_squared)*(len(y_trainm)-1))/((len(y_trainm)-X_trainm.shape[1]-1)))
    GS.at[i-5,'Adj. r2 of D + AET + SM vs. EVI'] = adj_r2
    
    #try relationship with test data
    y_predictm = model.predict(X_testm)
    
    #calculate MSE
    AICm = (2*X_trainm.shape[1]) - (2*np.log((sum((y_testm-y_predictm)**2))))
    GS.at[i-5,'AIC of D + AET + SM vs. EVI'] = AICm
    rMSEm = sqrt(mean_squared_error(y_testm, y_predictm))
    GS.at[i-5,'rMSE of D + AET + SM vs. EVI'] = rMSEm
    
    ### D + AET vs. EVI
    X_trainm2, X_testm2, y_trainm2, y_testm2 = train_test_split(EVI_p[['Sum of D','Sum of AET']], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)
    
    #estrablish multivariable linear relationship 
    model2 = LinearRegression()
    model2.fit(X_trainm2, y_trainm2)
    
    #calculated adjusted r2 value
    yhat2 = model2.predict(X_trainm2)
    SS_Residual2 = sum((y_trainm2-yhat2)**2)       
    SS_Total2 = sum((y_trainm2-np.mean(y_trainm2))**2)     
    r_squared2 = 1 - (float(SS_Residual2))/SS_Total2
    adj_r22 = 1 - (((1-r_squared2)*(len(y_trainm2)-1))/((len(y_trainm2)-X_trainm2.shape[1]-1)))
    GS.at[i-5,'Adj. r2 of D + AET vs. EVI'] = adj_r22
    
    #try relationship with test data
    y_predictm2 = model2.predict(X_testm2)
    
    #calculate MSE
    AICm2 = (2*X_trainm2.shape[1]) - (2*np.log((sum((y_testm2-y_predictm2)**2))))
    GS.at[i-5,'AIC of D + AET vs. EVI'] = AICm2
    rMSEm2 = sqrt(mean_squared_error(y_testm2, y_predictm2))
    GS.at[i-5,'rMSE of D + AET vs. EVI'] = rMSEm2
    
    ###  AET + SM vs. EVI
    X_trainm3, X_testm3, y_trainm3, y_testm3 = train_test_split(EVI_p[['Sum of AET','Average of SOIL']], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)
    
    #estrablish multivariable linear relationship 
    model3 = LinearRegression()
    model3.fit(X_trainm3, y_trainm3)
    
    #calculated adjusted r2 value
    yhat3 = model3.predict(X_trainm3)
    SS_Residual3 = sum((y_trainm3-yhat3)**2)       
    SS_Total3 = sum((y_trainm3-np.mean(y_trainm3))**2)     
    r_squared3 = 1 - (float(SS_Residual3))/SS_Total3
    adj_r23 = 1 - (((1-r_squared3)*(len(y_trainm3)-1))/((len(y_trainm3)-X_trainm3.shape[1]-1)))
    GS.at[i-5,'Adj. r2 of AET + SM vs. EVI'] = adj_r23
    
    #try relationship with test data
    y_predictm3 = model3.predict(X_testm3)
    
    #calculate MSE
    AICm3 = (2*X_trainm3.shape[1]) - (2*np.log((sum((y_testm3-y_predictm3)**2))))
    GS.at[i-5,'AIC of AET + SM vs. EVI'] = AICm3
    rMSEm3 = sqrt(mean_squared_error(y_testm3, y_predictm3))
    GS.at[i-5,'rMSE of AET + SM vs. EVI'] = rMSEm3
    
    
    ###  D + SM vs. EVI
    X_trainm4, X_testm4, y_trainm4, y_testm4 = train_test_split(EVI_p[['Sum of D','Average of SOIL']], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)
    
    #estrablish multivariable linear relationship 
    model4 = LinearRegression()
    model4.fit(X_trainm4, y_trainm4)
    
    #calculated adjusted r2 value
    yhat4 = model4.predict(X_trainm4)
    SS_Residual4 = sum((y_trainm4-yhat4)**2)       
    SS_Total4 = sum((y_trainm4-np.mean(y_trainm4))**2)     
    r_squared4 = 1 - (float(SS_Residual4))/SS_Total4
    adj_r24 = 1 - (((1-r_squared4)*(len(y_trainm4)-1))/((len(y_trainm4)-X_trainm4.shape[1]-1)))
    GS.at[i-5,'Adj. r2 of D + SM vs. EVI'] = adj_r24
    
    #try relationship with test data
    y_predictm4 = model4.predict(X_testm4)
    
    #calculate MSE
    AICm4 = (2*X_trainm4.shape[1]) - (2*np.log((sum((y_testm4-y_predictm4)**2))))
    GS.at[i-5,'AIC of D + SM vs. EVI'] = AICm4
    rMSEm4 = sqrt(mean_squared_error(y_testm, y_predictm))
    GS.at[i-5,'rMSE of D + SM vs. EVI'] = rMSEm4
    
    ###Figure out the best model fit for each period
    list = [r2D, r2A, r2S, adj_r2, adj_r22, adj_r23, adj_r24]
    best_r2 = list.index(max(list))
    cut = 0.05
    
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
        
        if((adj_r2 - r2D) <= cut or (adj_r2 - r2A) <= cut or (adj_r2 - r2S) <= cut):
            list2 = [r2D, r2A, r2S]
            best_r2ii = list2.index(max(list2))
    
            if best_r2ii == 0:
                Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
                Best.at[i-5,'r^2'] = r2D
                Best.at[i-5,'Intercept'] = interceptD
                Best.at[i-5,'Coefficient'] = slopeD
                Best.at[i-5,'AIC'] = AICD
                Best.at[i-5,'rMSE'] = rMSED
                
            
            elif best_r2ii ==1 :
                Best.at[i-5,'Best Predictor(s)'] = 'AET'
                Best.at[i-5,'r^2'] = r2A
                Best.at[i-5,'Intercept'] = interceptA
                Best.at[i-5,'Coefficient'] = slopeA
                Best.at[i-5,'AIC'] = AICA
                Best.at[i-5,'rMSE'] = rMSEA
                
            
            else:
                Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
                Best.at[i-5,'r^2'] = r2S
                Best.at[i-5,'Intercept'] = interceptS
                Best.at[i-5,'Coefficient'] = slopeS
                Best.at[i-5,'AIC'] = AICS
                Best.at[i-5,'rMSE'] = rMSES
               
        else:    
            Best.at[i-5,'Best Predictor(s)'] = 'D + AET + SM'
            Best.at[i-5,'adjusted r^2'] = adj_r2
            Best.at[i-5,'Intercept'] = model.intercept_
            Best.at[i-5,'Coefficient'] = model.coef_[0]
            Best.at[i-5,'Coefficient 2'] = model.coef_[1]
            Best.at[i-5,'Coefficient 3'] = model.coef_[2]
            Best.at[i-5,'AIC'] = AICm
            Best.at[i-5,'rMSE'] = rMSEm
           
            
            
                
    elif best_r2 == 4:
        
        if((adj_r22 - r2D) <= cut or (adj_r22 - r2A) <= cut or (adj_r22 - r2S) <= cut):
            list2 = [r2D, r2A, r2S]
            best_r2ii = list2.index(max(list2))
    
            if best_r2ii == 0:
                Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
                Best.at[i-5,'r^2'] = r2D
                Best.at[i-5,'Intercept'] = interceptD
                Best.at[i-5,'Coefficient'] = slopeD
                Best.at[i-5,'AIC'] = AICD
                Best.at[i-5,'rMSE'] = rMSED
                
            
            elif best_r2ii ==1 :
                Best.at[i-5,'Best Predictor(s)'] = 'AET'
                Best.at[i-5,'r^2'] = r2A
                Best.at[i-5,'Intercept'] = interceptA
                Best.at[i-5,'Coefficient'] = slopeA
                Best.at[i-5,'AIC'] = AICA
                Best.at[i-5,'rMSE'] = rMSEA
                
            
            else:
                Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
                Best.at[i-5,'r^2'] = r2S
                Best.at[i-5,'Intercept'] = interceptS
                Best.at[i-5,'Coefficient'] = slopeS
                Best.at[i-5,'AIC'] = AICS
                Best.at[i-5,'rMSE'] = rMSES
                
        else:    
            Best.at[i-5,'Best Predictor(s)'] = 'D + AET'
            Best.at[i-5,'adjusted r^2'] = adj_r22
            Best.at[i-5,'Intercept'] = model2.intercept_
            Best.at[i-5,'Coefficient'] = model2.coef_[0]
            Best.at[i-5,'Coefficient 2'] = model2.coef_[1]
            Best.at[i-5,'AIC'] = AICm2
            Best.at[i-5,'rMSE'] = rMSEm2
            
    
    elif best_r2 == 5:
        
        if((adj_r23 - r2D) <= cut or (adj_r23 - r2A) <= cut or (adj_r23 - r2S) <= cut):
            list2 = [r2D, r2A, r2S]
            best_r2ii = list2.index(max(list2))
    
            if best_r2ii == 0:
                Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
                Best.at[i-5,'r^2'] = r2D
                Best.at[i-5,'Intercept'] = interceptD
                Best.at[i-5,'Coefficient'] = slopeD
                Best.at[i-5,'AIC'] = AICD
                Best.at[i-5,'rMSE'] = rMSED
                
            
            elif best_r2ii ==1 :
                Best.at[i-5,'Best Predictor(s)'] = 'AET'
                Best.at[i-5,'r^2'] = r2A
                Best.at[i-5,'Intercept'] = interceptA
                Best.at[i-5,'Coefficient'] = slopeA
                Best.at[i-5,'AIC'] = AICA
                Best.at[i-5,'rMSE'] = rMSEA
                
            
            else:
                Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
                Best.at[i-5,'r^2'] = r2S
                Best.at[i-5,'Intercept'] = interceptS
                Best.at[i-5,'Coefficient'] = slopeS
                Best.at[i-5,'AIC'] = AICS
                Best.at[i-5,'rMSE'] = rMSES
                
        else:    
            Best.at[i-5,'Best Predictor(s)'] = 'AET + SM'
            Best.at[i-5,'adjusted r^2'] = adj_r23
            Best.at[i-5,'Intercept'] = model3.intercept_
            Best.at[i-5,'Coefficient'] = model3.coef_[0]
            Best.at[i-5,'Coefficient 2'] = model3.coef_[1]
            Best.at[i-5,'AIC'] = AICm3
            Best.at[i-5,'rMSE'] = rMSEm3
            
                
    elif best_r2 == 6:
        
        if((adj_r24 - r2D) <= cut or (adj_r24 - r2A) <= cut or (adj_r24 - r2S) <= cut):
            list2 = [r2D, r2A, r2S]
            best_r2ii = list2.index(max(list2))
    
            if best_r2ii == 0:
                Best.at[i-5,'Best Predictor(s)'] = 'Deficit (D)'
                Best.at[i-5,'r^2'] = r2D
                Best.at[i-5,'Intercept'] = interceptD
                Best.at[i-5,'Coefficient'] = slopeD
                Best.at[i-5,'AIC'] = AICD
                Best.at[i-5,'rMSE'] = rMSED
                
            
            elif best_r2ii ==1 :
                Best.at[i-5,'Best Predictor(s)'] = 'AET'
                Best.at[i-5,'r^2'] = r2A
                Best.at[i-5,'Intercept'] = interceptA
                Best.at[i-5,'Coefficient'] = slopeA
                Best.at[i-5,'AIC'] = AICA
                Best.at[i-5,'rMSE'] = rMSEA
                
            
            else:
                Best.at[i-5,'Best Predictor(s)'] = 'Soil Moisture (SM)'
                Best.at[i-5,'r^2'] = r2S
                Best.at[i-5,'Intercept'] = interceptS
                Best.at[i-5,'Coefficient'] = slopeS
                Best.at[i-5,'AIC'] = AICS
                Best.at[i-5,'rMSE'] = rMSES
                
        else:    
            Best.at[i-5,'Best Predictor(s)'] = 'D + SM'
            Best.at[i-5,'adjusted r^2'] = adj_r24
            Best.at[i-5,'Intercept'] = model4.intercept_
            Best.at[i-5,'Coefficient'] = model4.coef_[0]
            Best.at[i-5,'Coefficient 2'] = model4.coef_[1]
            Best.at[i-5,'AIC'] = AICm4
            Best.at[i-5,'rMSE'] = rMSEm4
    
#Create a blank dataframe that stores the best relationship slope intercept and r2 value and AIC





