# -*- coding: utf-8 -*-
"""
Created on Thu May 20 15:10:15 2021

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
Temp['4 Day Tmax MA']= Temp.iloc[:,3].rolling(window=4).mean()
Temp['4 Day Avg Temp MA'] = Temp.iloc[:,4].rolling(window=4).mean()
Tox['Date'] = pd.to_datetime(Tox['Date']).dt.tz_localize(None)
WT['Date'] = pd.to_datetime(WT['Date']).dt.tz_localize(None)

#Establish Linear relationship between Rolling average minimum Temp and water Temp
WTdf = pd.merge(Temp, WT, how='inner', on='Date')
RAT = WTdf['4 Day Tmin MA']
Ama = WTdf['4 Day Avg Temp MA']
WTmin = WTdf['Min W Temp (C)']
WTav = WTdf['Avg W Temp (C)']
mxT = WTdf['4 Day Tmax MA']
MXW = WTdf['Max W Temp (C)']

#function that calculates slope, intercept and R2 value.
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(RAT, WTmin)
print('4 Day Min Air Temp MA Vs. Min Water Temp')
print('R^2 = ',r_value**2)

#function that calculates slope, intercept and R2 value.
slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(Ama, WTav)
print('4 Day Avg Air Temp MA Vs. Avg Water Temp')
print('R^2 = ',r_value2**2)

#function that calculates slope, intercept and R2 value.
slope3, intercept3, r_value3, p_value3, std_err3 = scipy.stats.linregress(mxT, MXW)
print('4 Day Max Air Temp MA Vs. Max Water Temp')
print('R^2 = ',r_value3**2)
    
#combine all the datasets
VRdf = pd.merge(discharge,Temp, how='inner', on='Date')
VRdf = pd.merge(VRdf, WT, how='outer', on='Date')

#Add interpretted water temp data to VRdf
for k in range(7300,len(VRdf),1):
    WaterT = VRdf.at[k, 'Min W Temp (C)']
    RATm = VRdf.at[k, '4 Day Tmin MA']
    if np.isnan(WaterT):
        VRdf.at[k,'Min W Temp (C)'] = intercept + (slope*RATm)
    else:
        continue
    
for kb in range(7300,len(VRdf),1):
    WrT = VRdf.at[kb, 'Max W Temp (C)']
    RA = VRdf.at[kb, '4 Day Tmax MA']
    if np.isnan(WrT):
        VRdf.at[kb,'Max W Temp (C)'] = intercept3 + (slope3*RA)
    else:
        continue
    
for ka in range(7300,len(VRdf),1):
    WaT = VRdf.at[ka, 'Avg W Temp (C)']
    Avma = VRdf.at[ka, '4 Day Avg Temp MA']
    if np.isnan(WaT):
        VRdf.at[ka, 'Avg W Temp (C)'] = intercept2 + (slope2 *Avma)
    else:
        continue
    
#Calculate Daily Rate of Change of Water and 5 day average of rate of change
VRdf['Daily ROC Min WT'] = VRdf['Min W Temp (C)'].pct_change()
VRdf['5 Day Min Water Temp ROC (%)'] = abs(VRdf.iloc[:,15]).rolling(window=5).mean()
VRdf['Daily ROC Max WT'] = VRdf['Max W Temp (C)'].pct_change()
VRdf['5 Day Max Water Temp ROC (%)'] = abs(VRdf.iloc[:,17]).rolling(window=5).mean()
VRdf['Daily ROC Avg WT'] = VRdf['Avg W Temp (C)'].pct_change()
VRdf['5 Day Avg Water Temp ROC (%)'] = abs(VRdf.iloc[:,19]).rolling(window=5).mean()

#Dis = Size of Discharge event needed to reset GDD
#Lag = number of Days before resets after 

def AnatoxinPredictor(Dis = 150, Lag = 2, GDDmin = 7):
    
    #establish empty 
    VRdf['Count'] = np.nan
    
    for i in range (7305,len(VRdf),1):
        
        cntii = 0
        test = False
        while test == False:  
            for j in range(i,0,-1):
                
                Hdis = VRdf.at[j,'Discharge (cfs)']
                
                if Hdis < 150:
                
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



    VRdf['GDD'] = np.nan
    
    for g in range (7305,len(VRdf),1):
        
        AVWT = VRdf.at[g, 'Avg W Temp (C)']
        CNN = VRdf.at[g, 'Count']
        
        if AVWT <= GDDmin or CNN <= Lag:
            
            VRdf.at[g,'GDD'] = 0
            
        else:
            VRdf.at[g,'GDD'] = AVWT - GDDmin
        

#Print Datedf to find min and max values of climatic indicators in excel
Datedf = pd.merge(Datedf, VRdf, how ='inner', on='Date')
#Datedf.to_csv('Thresholds.csv')

VRdf = pd.merge(VRdf, Tox, how='outer', on='Date' )

#print to csv
VRdf.to_csv('VRdf.csv')

#concatenate with just Tox Data
toxdf = pd.merge(VRdf, Tox, how ='inner', on='Date') 

toxdf['Concentration'] = toxdf['Anatoxin-a concentration'].astype(float)
toxdf['Water'] = toxdf['Min W Temp (C)'].astype(float)

#Add a column of Bloom Vs. No Bloom
toxdf['Bloom'] = ''
toxdf['Bloom'] = toxdf['Bloom'].astype(str)

for ii in range(0,len(toxdf),1):
    if toxdf.at[ii,'Concentration']>=15:
        toxdf.at[ii, 'Bloom'] = 'Yes'
    else:
        toxdf.at[ii, 'Bloom'] ='No'


# with quantile regression
#fit the model
model = smf.quantreg('Concentration ~ Water',toxdf).fit(q=0.95)
model2 = smf.quantreg('Concentration ~ Water',toxdf).fit(q=0.80)
#model3 = smf.quantreg('Concentration ~ Water',toxdf).fit(q=0.925)

#define figure and axis
fig, ax = plt.subplots(figsize=(8, 6))

#get y values
get_y = lambda a, b: a + b * toxdf.Water
y = get_y(model.params['Intercept'], model.params['Water'])
y2 = get_y(model2.params['Intercept'], model.params['Water'])
#y3 = get_y(model3.params['Intercept'], model.params['Water'])

#plot data points with quantile regression equation overlaid
ax.plot(toxdf.Water, y, color='black')
ax.plot(toxdf.Water, y2, color='red')
#ax.plot(toxdf.Water, y3, color='green')
ax.scatter(toxdf.Water, toxdf.Concentration, alpha=.5, color ='blue')
ax.set_xlabel('Water Temperature (C)', fontsize=14)
ax.set_ylabel('Anatoxin-a Concentration (ug/L)', fontsize=14)

toxdf['Discharge (cfs)'] = toxdf['Discharge (cfs)'].astype(float)
toxdf['Count'] = toxdf['Count'].astype(float)
toxdf['5 Day Min Water Temp ROC (%)'] = toxdf['5 Day Min Water Temp ROC (%)'].astype(float)
toxdf['10 Day Discharge MA'] = toxdf['10 Day Discharge MA'].astype(float)









