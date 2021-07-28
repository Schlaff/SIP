# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:45:16 2021

@author: brian
"""

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats


#import discharge, Temperature, and water Temp data
discharge = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Discharge\north_fork_virgin_stream.csv')
Temp = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Historic_P&T\All_years_Final_CanyonJunction.csv')
WT = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Water_Temp\WT_DNR.csv')


#Delete Extraneous Columns in dataframes
discharge = discharge[['Date','Discharge (cfs)']]
Temp = Temp[['Date','Precip (mm)','Min Temp (C)','Max Temp (C)','Avg Temp (C)']]


#replace 
discharge = discharge.replace(r'^\s*$', np.nan, regex=True)

#Drop columns with NA values
discharge.dropna(subset = ["Discharge (cfs)"], inplace=True)
WT.dropna(subset = ['Avg W Temp (C)'], inplace = True)

#Change Variable type and calculate daily rate of change of discharge
discharge['Discharge (cfs)'] = pd.to_numeric(discharge['Discharge (cfs)'])
discharge['Date'] = pd.to_datetime(discharge['Date']).dt.tz_localize(None)
Temp['Date'] = pd.to_datetime(Temp['Date']).dt.tz_localize(None)


#Add ROC and MA data
discharge['Daily Discharge ROC (%)'] = discharge['Discharge (cfs)'].pct_change(axis=0)
discharge['10 Day Discharge MA'] = discharge.iloc[:,1].rolling(window=10).mean()
discharge['10 Day ROC MA'] = abs(discharge.iloc[:,2]).rolling(window=10).mean()
Temp['4 Day Tmin MA'] = Temp.iloc[:,2].rolling(window=4).mean()
WT['Date'] = pd.to_datetime(WT['Date']).dt.tz_localize(None)

#Establish Linear relationship between Rolling average minimum Temp and water Temp
WTdf = pd.merge(Temp, WT, how='inner', on='Date')
RAT = WTdf['4 Day Tmin MA']
WTmin = WTdf['Min W Temp (C)']

#function that calculates slope, intercept and R2 value.
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(RAT, WTmin)
print('4 Day Min Air Temp MA Vs. Min Water Temp')
print('R^2 = ',r_value**2)
    
#combine all the datasets
VRdf = pd.merge(discharge,Temp, how='inner', on='Date')
VRdf = pd.merge(VRdf, WT, how='outer', on='Date')


#Add interpretted water temp data to VRdf
for k in range(7305,len(VRdf),1):
    WaterT = VRdf.iloc[k]['Min W Temp (C)']
    RATm = VRdf.iloc[k]['4 Day Tmin MA']
    if np.isnan(WaterT):
        VRdf['Min W Temp (C)'][k] = intercept + (slope*RATm)
    else:
        continue
    
#Calculate Daily Rate of Change of Water and 5 day average of rate of change
VRdf['Daily Water Temp ROC (%)'] = VRdf['Min W Temp (C)'].pct_change()
VRdf['5 Day Water Temp ROC (%)'] = abs(VRdf.iloc[:,13]).rolling(window=5).mean()

#loop through and count the amount of days of low discharge events (<49)
BloomDays = {} 
cnt=0
ttl=len(VRdf)

for i in range(7305,len(VRdf),1):
    
    
    #Establish Variables
    key = VRdf.iloc[i]['Date']
    Dis = VRdf.iloc[i]['Discharge (cfs)']
    ROC = abs(VRdf.iloc[i]['Daily Discharge ROC (%)'])
    AvDis = VRdf.iloc[i]['10 Day Discharge MA']
    AvROC = VRdf.iloc[i]['10 Day ROC MA']
    Tmin = VRdf.iloc[i]['Min Temp (C)']
    AvTmin = VRdf.iloc[i]['4 Day Tmin MA']
    Wat = VRdf.iloc[i]['Min W Temp (C)']
    WatROC = VRdf.iloc[i]['5 Day Water Temp ROC (%)']
    
    # While Loop to count amount of days since discharge >60

    cntii = 0
    test = False
    
    while test == False:  
        for j in range(i,0,-1):
            
            Hdis = VRdf.iloc[j]['Discharge (cfs)']
            
            if Hdis < 60:
            
                cntii += 1
                
                if j == 0:
                    
                    test = True
                    break
                
                else:
                    j += -1
                    
            else:
                test = True  
                break
        
            
    #Check if date listed had climatic conditions within the range of the days Prior blooms happened  
    if cntii >= 25 and Dis <=48 and AvDis <= 50.2 and AvROC <= 0.065 and AvTmin >= (-2.53) and Wat >=5.82 and WatROC <= 0.116:
        
    
        cnt += 1   
        BloomDays.update({cnt:key})
    
    print(((i-7305)/(ttl-7305))*100,'%')
    
#Create a new dataframe with averaged canopy area
BloomDaysDF = pd.DataFrame.from_dict(BloomDays, orient= 'index')

#Check which days are predicted for Bloom and which days are not

