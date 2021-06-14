# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 15:44:17 2021

@author: brian
"""

import csv
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#

airT = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Halls Creek\All_years_Final_Halls.csv')
watT = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Halls Creek\CARE_Halls_tempw.csv')

airT= airT[['Date','Min Temp (C)','Max Temp (C)','Avg Temp (C)']]


#for i in range (1,11,1):
    
   # airT['4 Day Tmin MA'] = airT.iloc[:,1].rolling(window= i ).mean()
   # airT['4 Day Tmax MA'] = airT.iloc[:,2].rolling(window= i).mean()
   # airT['4 Day Avg MA']= airT.iloc[:,3].rolling(window= i).mean()
   # WTdf = pd.merge(airT, watT, how='inner', on='Date')
    
   # mnT = WTdf['4 Day Tmin MA']
   # avT = WTdf['4 Day Avg MA']
   # mxT = WTdf['4 Day Tmax MA']
   # W = WTdf['Temp']
    
   # print('')
   # print(i,'Day Moving Average')
    #function that calculates slope, intercept and R2 value.
   # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(mnT, W)
   # print(i,'Day Min Air Temp MA Vs. Water Temp')
   # print('R^2 = ',r_value**2)

#function that calculates slope, intercept and R2 value.
   # slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(avT, W)
   # print(i,'Day Avg Air Temp MA Vs. Water Temp')
   # print('R^2 = ',r_value2**2)

#function that calculates slope, intercept and R2 value.
   # slope3, intercept3, r_value3, p_value3, std_err3 = scipy.stats.linregress(mxT, W)
   # print(i,'Day Max Air Temp MA Vs. Water Temp')
   # print('R^2 = ',r_value3**2)


Best = pd.merge(airT, watT, how='inner', on='Date')
x = Best['Avg Temp (C)']
y = Best['Temp']
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
print('Slope=',slope,' intercept=', intercept)
print('R^2 = ',r_value**2)
print('p valaue=', p_value)


# Plot     
plt.xlabel('Avg Air Temp (C)')
plt.ylabel('Water Temp (C)')
plt.scatter(x, y, marker='o')
m, b = np.polyfit(x, y, 1)
plt.plot(x, m*x + b)
