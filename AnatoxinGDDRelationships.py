# -*- coding: utf-8 -*-
"""
This script is to predict Anatoxin-a Concentration on the North Fork  
of the Virgin River between The Narrows and the Southern Entry of Zion 
National Park.  

Created on Fri Jun 11 11:20:51 2021

@author: Brian schlaff
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
from sklearn.metrics import r2_score


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

#replace missing values with Na in Discharge Data Frame
discharge = discharge.replace(r'^\s*$', np.nan, regex=True)

#Drop rows with NA values in Discharge and 
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

####Establish Linear relationship between Rolling average minimum Temp and water Temp
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

####Fill in missing water data With predictive relationships established
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


#Lag = Amount of Days between GDD and Anatoxin response
#Dis = Size of Discharge event needed to reset GDD
#Length = number of Days before resets after a large discharge event
#GDDmin = Theshold value of GDD


def Anatoxin_Predict(Lag=2, Dis = 150, Length = 3, GDDmin = 7):
    
    #Combine Toxin Data set with VRdf
    VRtox = pd.merge(VRdf, Tox, how='outer', on='Date' )
    
    #establish empty 
    VRtox['Count'] = np.nan
    
    for i in range (7305,len(VRtox),1):
        
        cntii = 0
        test = False
        while test == False:  
            for j in range(i,0,-1):
                
                Hdis = VRtox.at[j,'Discharge (cfs)']
                
                if Hdis < 150:
                
                    cntii += 1
                    
                    if j == 0:
                        
                        test = True
                        break
                    
                    else:
                        j += -1
                        
                else:
                    VRtox.at[i, 'Count'] = cntii
                    test = True  
                    break



    VRdf['GDD'] = np.nan
    
    for g in range (7305,len(VRdf),1):
        
        AVWT = VRtox.at[g, 'Avg W Temp (C)']
        CNN = VRtox.at[g, 'Count']
        
        if AVWT <= GDDmin or CNN <= Length:
            
            VRtox.at[g,'GDD'] = 0
            
        else:
            VRtox.at[g,'GDD'] = AVWT - GDDmin
            
    
    
    #shift column number of days established in definition
    VRtox['Anatoxin-a concentration'] = VRtox['Anatoxin-a concentration'].shift(0-Lag)

    
    #Plot GDD and Anatoxin-a relationships over time
    A = VRtox['Anatoxin-a concentration']
    G = VRtox['GDD']
    D = VRtox['Date']
    
    
    ##Plot GDD and Anatoxin-a Concentrations Overtime
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(D, G, color="green")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=12)
    # set y-axis label
    ax.set_ylabel("GDD (C)",color="green",fontsize=12)
    ax.set_xlim([dt.date(2020, 1, 1), max(D)])
    
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(D, A,color="red",marker="o")
    ax2.set_ylabel("Anatoxin-a Concentration (ug/L)",color="red",fontsize=12)
    plt.show()
    
    #Calculate relationship between GDD and Anatoxin-a 
    Bloom = pd.DataFrame(columns =['Date','GDD','Anatoxin-a concentration'])
    Bloom = VRtox[['Date','GDD','Anatoxin-a concentration']]
    Bloom.dropna(subset = ["Anatoxin-a concentration"], inplace=True)
    
    AA = Bloom['Anatoxin-a concentration']
    GG = Bloom['GDD']
    
    slope4, intercept4, r_value4, p_value4, std_err4 = scipy.stats.linregress(GG, AA)
    r2 = round((r_value4**2),4)
    
    xx = [0, max(GG)]
    yy = [intercept4,(((18)*slope4)+intercept4)]
    
    ##
    plt.figure(2)
    plt.ylabel('Anatoxin-a Concentration (ug/L)')
    plt.xlabel('GDD (C)')
    plt.ylim([-5,700])
    plt.xlim([0,18])
    plt.scatter(GG, AA, color='blue', marker='o')
    plt.plot(xx, yy, color = 'black')
    plt.annotate('r^2 =', (1,500))
    plt.annotate(r2,(3, 500))