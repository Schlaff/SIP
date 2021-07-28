# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 15:12:05 2021

@author: brian
"""

import csv
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

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

#create blank dataframe of just growing season data
GS = pd.DataFrame()
Period = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
Dates = ['March 6 - March 21', 'March 22 - April 6', 'April 7 - April 22', 'April 23 - May 8', 'May 9 - May 24', 'May 25 - June 9', 'June 10 - June 25', 'June 26 - July 11', 'July 12 - July 27', 'July 28 - August 12', 'August 13 - August 28', 'August 29 - September 13', 'September 14 - September 29', 'September 30 - October 14', 'October 15 - October 31']

#Create a blank dataframe that stores the best relationship slope intercept and r2 value
Best = pd.DataFrame()
Best['Date Range'] = Dates
Best['Period'] = Period
Best['Best Predictor(s)'] = ''
Best['r^2'] = np.nan
Best['rMSE'] = np.nan

# loop through the data and Calculate R2 and MSE for all of the different periods

for i in range(5,20,1):
    print('')
    print('/////////////////////')
    print('Period', i)
    EVI_p = EVI[EVI["Period"] == i]
    
    #split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(EVI_p[['Sum of P','Sum of MELT','Sum of D','Sum of AET','Average of SOIL']], EVI_p['scaled EVI3'], test_size=0.25, random_state=12)
    
    #Set up KFold 
    folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
    
    #specify hyperparameters to tune
    hyper_params = [{'n_features_to_select': list(range(1, 6))}]
    
    #set up linear regression model and RFE
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    rfe = RFE(lm)             
    
    #perform gridsearch to select best predictors
    model_cv = GridSearchCV(estimator = rfe, 
                            param_grid = hyper_params, 
                            scoring= 'neg_root_mean_squared_error', 
                            cv = folds, 
                            verbose = 1,
                            return_train_score=True)      
    
    #fit the model
    model_cv.fit(X_train, y_train)    
    
    #cross validation results
    cv_results = pd.DataFrame(model_cv.cv_results_)
    cv_results
    
    #plot cross validation results
    plt.figure(figsize=(16,6))
    
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
    plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
    plt.xlabel('number of features')
    plt.ylabel('r-squared')
    plt.title("Optimal Number of Features")
    plt.legend(['test score', 'train score'], loc='upper left')
    
    #select number of features
    MeanTS = cv_results["mean_test_score"]
    max_index = MeanTS.idxmax()
    
    n_features_optimal = cv_results.at[max_index,"param_n_features_to_select"]
    
    lm = LinearRegression()
    lm.fit(X_train, y_train)
    
    rfe = RFE(lm, n_features_to_select=n_features_optimal)             
    rfe = rfe.fit(X_train, y_train)
    
    # summarize all features
    print('Best Predictors')
    for j in range(X_train.shape[1]):
        if j == 0 and rfe.support_[j] == True:
            Best.at[i-5,'Best Predictor(s)'] = 'Precipitation,'
            print('Rain')
        elif j == 1 and rfe.support_[j] == True:
            Best.at[i-5,'Best Predictor(s)'] = Best.at[i-5,'Best Predictor(s)'] +  ' Snow Melt,'
            print('Snow Melt')
        elif j == 2 and rfe.support_[j] == True:
            Best.at[i-5,'Best Predictor(s)'] = Best.at[i-5,'Best Predictor(s)'] +  ' Deficit,'
            print('Deficit')
        elif j == 3 and rfe.support_[j] == True:
            Best.at[i-5,'Best Predictor(s)'] = Best.at[i-5,'Best Predictor(s)'] +  ' AET,'
            print('AET')
        elif j == 4 and rfe.support_[j] == True:
            Best.at[i-5,'Best Predictor(s)'] = Best.at[i-5,'Best Predictor(s)'] +  ' Soil Moisture,'
            print('Soil Moisture')
    	
    
    # predict prices of X_test
    y_pred = lm.predict(X_test)
    y_proj = lm.predict(X_train)
    r2 = r2_score(y_train, y_proj)
    print('r^2 =',r2)
    rMSE = sqrt(mean_squared_error(y_test, y_pred))
    print('rMSE=', rMSE)
    Best.at[i-5,'r^2'] = r2
    Best.at[i-5,'rMSE'] = rMSE
    Best.at[i-5,'Intercept'] = lm.intercept_
    Best.at[i-5,'Coefficient'] = lm.coef_[0]
    Best.at[i-5,'Coefficient 2'] = lm.coef_[1]
    Best.at[i-5,'Coefficient 3'] = lm.coef_[2]
    