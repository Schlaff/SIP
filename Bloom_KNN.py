# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 18:27:37 2021

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
from pygam import LinearGAM
from sklearn.neighbors import KNeighborsClassifier

#import discharge, Temperature, and water Temp data
discharge = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Discharge\north_fork_virgin_stream.csv')
Temp = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Historic_P&T\All_years_Final_CanyonJunction.csv')
WT = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Water_Temp\WT_DNR.csv')
Tox = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\anatoxin concentration.csv')
Datedf = pd.read_csv(r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\BloomDates.csv')


#Delete Extraneous Columns in dataframes
discharge = discharge[['Date','Discharge (cfs)']]
Temp = Temp[['Date','Precip (mm)','Min Temp (C)','Max Temp (C)','Avg Temp (C)']]
Tox = Tox[['Date','Anatoxin-a concentration']]

#replace 
discharge = discharge.replace(r'^\s*$', np.nan, regex=True)

#Drop columns with NA values
discharge.dropna(subset = ["Discharge (cfs)"], inplace=True)
WT.dropna(subset = ['Avg W Temp (C)'], inplace = True)

#Change Variable type and calculate daily rate of change of discharge
discharge['Discharge (cfs)'] = pd.to_numeric(discharge['Discharge (cfs)'])
discharge['Date'] = pd.to_datetime(discharge['Date']).dt.tz_localize(None)
Temp['Date'] = pd.to_datetime(Temp['Date']).dt.tz_localize(None)
Tox['Date'] = pd.to_datetime(Tox['Date']).dt.tz_localize(None)
Datedf['Date'] = pd.to_datetime(Datedf['Date']).dt.tz_localize(None)

#only keep max values of anatoxin-a concentration by date
Tox = Tox.groupby(['Date'], as_index=False).max()

#Add ROC and MA data
discharge['Daily Discharge ROC (%)'] = discharge['Discharge (cfs)'].pct_change(axis=0)
discharge['10 Day Discharge MA'] = discharge.iloc[:,1].rolling(window=10).mean()
discharge['10 Day ROC MA'] = abs(discharge.iloc[:,2]).rolling(window=10).mean()
Temp['4 Day Tmin MA'] = Temp.iloc[:,2].rolling(window=4).mean()
#Temp['Daily Min T ROC (%)'] = Temp['Min Temp (C)'].pct_change()
#Temp['Daily Max T ROC (%)'] = Temp['Max Temp (C)'].pct_change()
#Temp['5 Day Min T ROC (%)'] = abs(Temp.iloc[:,6]).rolling(window=5).mean()
#Temp['5 Day Max T ROC (%)'] = abs(Temp.iloc[:,7]).rolling(window=5).mean()
Tox['Date'] = pd.to_datetime(Tox['Date']).dt.tz_localize(None)
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
VRdf['Daily Min Water Temp ROC (%)'] = VRdf['Min W Temp (C)'].pct_change()
VRdf['5 Day Min Water Temp ROC (%)'] = abs(VRdf.iloc[:,13]).rolling(window=5).mean()

VRdf['Count'] = np.nan

for i in range (7305,len(VRdf),1):
    
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
                VRdf.at[i, 'Count'] = cntii
                test = True  
                break
        
    

#Print Datedf to find min and max values of climatic indicators in excel
Datedf = pd.merge(Datedf, VRdf, how ='inner', on='Date')
#Datedf.to_csv('Thresholds.csv')

#concatenate with just Tox Data
VRdf = pd.merge(VRdf, Tox, how ='outer', on='Date') 

#Add a bloom Column
VRdf['Bloom'] = ''
VRdf['Bloom'] = VRdf['Bloom'].astype(str)
VRdf.to_csv('VRdf.csv')

for iii in range(7305,len(VRdf),1):
    if VRdf.iloc[iii]['Anatoxin-a concentration']>=15:
        VRdf.at[iii, 'Bloom'] = 'Yes'
    elif VRdf.iloc[iii]['Anatoxin-a concentration']>=0:
        VRdf.at[iii, 'Bloom'] ='NO'
    else:
        VRdf.at[iii, 'Bloom'] =''


###Classify Bloom Vs No Bloom with KNN
train = VRdf.iloc[[14800,14806,14811,14820,14834,14862,14878,14883,14912,14914,14931,14932,14959,14960,14983,15044],]
test = VRdf.iloc[7310:15099,]
X = ['Discharge (cfs)','10 Day Discharge MA','10 Day ROC MA','Min W Temp (C)','5 Day Min Water Temp ROC (%)','Count']
Y = ['Bloom']

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train[X], train[Y])
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(train[X], train[Y])))

pred = pd.DataFrame(knn.predict(test[X]))

