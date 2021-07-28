# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 11:25:40 2021

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
import pyflux as pf
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter



#import EVI data
EVI = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Sandy_3_Allotment\S3_WB_EVI.csv')

#change date to datetime 
EVI['date'] = pd.to_datetime(EVI['date']).dt.tz_localize(None)

#drop NA values
EVI.dropna(subset = ['scaled EVI3'], inplace = True)

#check the correlation of variables
f = plt.figure(figsize=(19, 15))
corr = EVI.corr()
plt.matshow(corr, fignum=f.number)
plt.xticks(range(EVI.select_dtypes(['number']).shape[1]), EVI.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(EVI.select_dtypes(['number']).shape[1]), EVI.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16);

#cleanup dataframe to only uncorrelated variables of interest
EVI['Deficit'] = EVI['Sum of D']
EVI['AET'] = EVI['Sum of AET']
EVI['SM'] =EVI['Average of SOIL']
EVI['Melt'] = EVI['Sum of MELT']
EVI['W'] = EVI['Sum of W']
EVI['EVI'] = EVI['scaled EVI3']
EVI=EVI[['date','year','Period','year_period','Melt','W','Deficit','AET','SM','EVI']]
EVI['EVI'] = savgol_filter(EVI.EVI, 7, 2)
EVI = EVI[(5<=EVI.Period) & (EVI.Period<=19)]

#Plot the EVI data overtime
plt.figure(figsize=(15,5));
plt.plot(EVI['date'],EVI['EVI'])
plt.ylabel('EVI')
plt.title('Modis Derived EVI 2016-2021')
plt.plot()

Arimax = pd.DataFrame()
Arimax['Model'] = ['D','AET','SM','D + AET + SM','D + AET','AET + SM', 'D + SM', 'M', 'M + D', 'M + AET', 'M + SM', 'M + D + AET', 'M + D + SM', 'M + AET + SM', 'M + D + AET + SM', 'W', 'W + AET']


#split the data into training and test data
for i in range(0,1,1):
    z = i

    #Split data into test and training data
    num_of_rows = round(len(EVI)*0.75)
    train = EVI.iloc[:num_of_rows] 
    test = EVI.iloc[num_of_rows:] 
    y_test = test[['EVI']]
    
    h = len(test)
    p = int(len(train)/2)
    
    ###build the model
    modelD = pf.ARIMAX(data=train, formula='EVI~ Deficit',
                      ar=z, ma=z, family=pf.Normal())
    
    #fit the model and get a summary
    modD = modelD.fit("MLE")
    modD.summary()
    
    #plot the model and the prediction for 2016-2021
    plot1 = modelD.plot_fit(figsize=(15,5))
    plot1_predict = modelD.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    #predict the output for 2016-2021 and check the rMSE
    y_predD = modelD.predict(h,test,intervals = False,)
    rMSED = sqrt(mean_squared_error(y_predD['EVI'], y_test))
    Arimax.at[0, i+1] = rMSED
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_predD['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    
    ###build the model
    modelA = pf.ARIMAX(data=train, formula='EVI~ AET',
                      ar=z, ma=z, family=pf.Normal())
    
    #fit the model and get a summary
    modA = modelA.fit("MLE")
    modA.summary()
    
    #plot the model and the prediction for 2016-2021
    plot2 = modelA.plot_fit(figsize=(15,5))
    plot2_predict = modelA.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    #predict the output for 2016-2021 and check the rMSE
    y_predA = modelA.predict(h,test,intervals = False,)
    rMSEA = sqrt(mean_squared_error(y_predA['EVI'], y_test))
    Arimax.at[1, i+1] = rMSEA
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_predA['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ###build the model
    modelS = pf.ARIMAX(data=train, formula='EVI~ SM',
                      ar=z, ma=z, family=pf.Normal())
    
    #fit the model and get a summary
    modS = modelS.fit("MLE")
    modS.summary()
    
    #plot the model and the prediction for 2016-2021
    plot3 = modelS.plot_fit(figsize=(15,5))
    plot3_predict = modelS.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    
    #predict the output for 2016-2021 and check the rMSE
    y_predS = modelS.predict(h,test,intervals = False,)
    rMSES = sqrt(mean_squared_error(y_predS['EVI'], y_test))
    Arimax.at[2, i+1] = rMSES
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_predS['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ###build the model
    model = pf.ARIMAX(data=train, formula='EVI~ Deficit + AET + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    #fit the model and get a summary
    mod = model.fit("MLE")
    mod.summary()
    
    #plot the model and the prediction for 2016-2021
    plot4 = model.plot_fit(figsize=(15,5))
    plot4_predict = model.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    #predict the output for 2016-2021 and check the rMSE
    y_pred = model.predict(h,test,intervals = False,)
    rMSE = sqrt(mean_squared_error(y_pred['EVI'], y_test))
    Arimax.at[3, i+1] = rMSE
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model1 = pf.ARIMAX(data=train, formula='EVI~Deficit + AET',
                      ar=z, ma=z, family=pf.Normal())
    
    mod1 = model1.fit("MLE")
    mod1.summary()
    
    plot5 = model1.plot_fit(figsize=(15,5))
    plot5_predict = model1.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    y_pred1 = model1.predict(h,test,intervals = False,)
    rMSE1 = sqrt(mean_squared_error(y_pred1['EVI'], y_test))
    Arimax.at[4, i+1] = rMSE1
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred1['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model2 = pf.ARIMAX(data=train, formula='EVI~AET + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod2 = model2.fit("MLE")
    mod2.summary()
    
    plot6 =model2.plot_fit(figsize=(15,5))
    plot6_predict = model2.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    y_pred2 = model2.predict(h,test,intervals = False,)
    rMSE2 = sqrt(mean_squared_error(y_pred2['EVI'], y_test))
    Arimax.at[5, i+1] = rMSE2
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred2['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model3 = pf.ARIMAX(data=train, formula='EVI~Deficit + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod3 = model3.fit("MLE")
    mod3.summary()
    
    plot7 = model3.plot_fit(figsize=(15,5))
    plot7_predict = model3.plot_predict(h = h, past_values = p, intervals = False, oos_data = test)
    
    y_pred3 = model3.predict(h,test,intervals = False,)
    rMSE3 = sqrt(mean_squared_error(y_pred3['EVI'], y_test))
    Arimax.at[6, i+1] = rMSE3
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred3['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model4 = pf.ARIMAX(data=train, formula='EVI~Melt',
                      ar=z, ma=z, family=pf.Normal())
    
    mod4 = model4.fit("MLE")
    mod4.summary()
    
    plot8 = model4.plot_fit(figsize=(15,5))
    
    y_pred4 = model4.predict(h,test,intervals = False,)
    rMSE4 = sqrt(mean_squared_error(y_pred4['EVI'], y_test))
    Arimax.at[7, i+1] = rMSE4
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred4['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
     ### new model with different exogenous variables
    model5 = pf.ARIMAX(data=train, formula='EVI~Melt + Deficit',
                      ar=z, ma=z, family=pf.Normal())
    
    mod5 = model5.fit("MLE")
    mod5.summary()
    
    plot9 = model5.plot_fit(figsize=(15,5))
    
    y_pred5 = model5.predict(h,test,intervals = False,)
    rMSE5 = sqrt(mean_squared_error(y_pred5['EVI'], y_test))
    Arimax.at[8, i+1] = rMSE5
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred5['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model6 = pf.ARIMAX(data=train, formula='EVI~Melt + AET',
                      ar=z, ma=z, family=pf.Normal())
    
    mod6 = model6.fit("MLE")
    mod6.summary()
    
    plot6 = model6.plot_fit(figsize=(15,5))
    
    y_pred6 = model6.predict(h,test,intervals = False,)
    rMSE6 = sqrt(mean_squared_error(y_pred6['EVI'], y_test))
    Arimax.at[9, i+1] = rMSE6
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred6['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model7 = pf.ARIMAX(data=train, formula='EVI~Melt + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod7 = model7.fit("MLE")
    mod7.summary()
    
    plot10 = model7.plot_fit(figsize=(15,5))
    
    y_pred7 = model7.predict(h,test,intervals = False,)
    rMSE7 = sqrt(mean_squared_error(y_pred7['EVI'], y_test))
    Arimax.at[10, i+1] = rMSE7
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred7['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model8 = pf.ARIMAX(data=train, formula='EVI~Melt + Deficit + AET',
                      ar=z, ma=z, family=pf.Normal())
    
    mod8 = model8.fit("MLE")
    mod8.summary()
    
    plot11 = model8.plot_fit(figsize=(15,5))
    
    y_pred8 = model8.predict(h,test,intervals = False,)
    rMSE8 = sqrt(mean_squared_error(y_pred8['EVI'], y_test))
    Arimax.at[11, i+1] = rMSE8
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred8['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model9 = pf.ARIMAX(data=train, formula='EVI~Melt + Deficit + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod9 = model9.fit("MLE")
    mod9.summary()
    
    plot12 = model9.plot_fit(figsize=(15,5))
    
    y_pred9 = model9.predict(h,test,intervals = False,)
    rMSE9 = sqrt(mean_squared_error(y_pred9['EVI'], y_test))
    Arimax.at[12, i+1] = rMSE9
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred9['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model10 = pf.ARIMAX(data=train, formula='EVI~Melt + AET + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod10 = model10.fit("MLE")
    mod10.summary()
    
    plot13 = model10.plot_fit(figsize=(15,5))
    
    y_pred10 = model10.predict(h,test,intervals = False,)
    rMSE10 = sqrt(mean_squared_error(y_pred10['EVI'], y_test))
    Arimax.at[13, i+1] = rMSE10
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred10['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model11 = pf.ARIMAX(data=train, formula='EVI~Melt + Deficit + AET + SM',
                      ar=z, ma=z, family=pf.Normal())
    
    mod11 = model11.fit("MLE")
    mod11.summary()
    
    plot14 = model11.plot_fit(figsize=(15,5))
    
    y_pred11 = model11.predict(h,test,intervals = False,)
    rMSE11 = sqrt(mean_squared_error(y_pred11['EVI'], y_test))
    Arimax.at[14, i+1] = rMSE11
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred11['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model12 = pf.ARIMAX(data=train, formula='EVI~W',
                      ar=z, ma=z, family=pf.Normal())
    
    mod12 = model12.fit("MLE")
    mod12.summary()
    
    plot15 = model12.plot_fit(figsize=(15,5))
    
    y_pred12 = model12.predict(h,test,intervals = False,)
    rMSE12 = sqrt(mean_squared_error(y_pred12['EVI'], y_test))
    Arimax.at[15, i+1] = rMSE12
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred12['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()
    
    ### new model with different exogenous variables
    model13 = pf.ARIMAX(data=train, formula='EVI~ AET + W',
                      ar=z, ma=z, family=pf.Normal())
    
    mod13 = model13.fit("MLE")
    mod13.summary()
    
    plot16 = model13.plot_fit(figsize=(15,5))
    
    y_pred13 = model13.predict(h,test,intervals = False,)
    rMSE13 = sqrt(mean_squared_error(y_pred13['EVI'], y_test))
    Arimax.at[16, i+1] = rMSE13
    
    #Plot overlayed predicted and test data
    plt.figure(figsize=(15,5));
    plt.plot(test['date'],y_test)
    plt.plot(test['date'],y_pred13['EVI'])
    plt.legend(['Test EVI','Predicted EVI'])
    plt.ylabel('EVI')
    plt.title('EVI 2016-2021')
    plt.plot()

Arimax['rMSE'] = Arimax[1]
Arimax = Arimax[['Model','rMSE']]





