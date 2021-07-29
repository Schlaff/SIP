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
import math
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
turb = pd.read_csv (r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Turbidity\Turbidity.csv')


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
turb['Date'] = pd.to_datetime(turb['Date']).dt.date
turb['Turbidity'] = turb['Turbidity'].astype(float)

#Sum turbidity data by date
turb = turb.groupby(['Date'], as_index=False).sum()

#then change date time value
turb['Date'] = pd.to_datetime(turb['Date']).dt.tz_localize(None)

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

###Plot 
x=[min(Ama), (intercept2 + ((min(Ama))*slope2))]
y=[max(Ama), (intercept2 + ((max(Ama))*slope2))]

plt.figure(2)
plt.ylabel('Average Water temperature')
plt.xlabel('4-day average air temperature')
plt.scatter(Ama, WTav, color='blue', marker='o')
plt.annotate('r^2 =', (1,20))
plt.annotate(r_value2**2,(5, 20))


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

#Merge Turbidity dataset with VRDF
VRdf = pd.merge(VRdf, turb, how='outer', on='Date')


#Lag = Amount of Days between GDD and Anatoxin response
#Dis = Size of Discharge event needed to reset GDD
#Length = number of Days before resets after a large discharge event
#Turb = Turbidity that will reset GDD to 0
#Length_t = number of Days before resets after a large discharge event
#GDDmin = Theshold value of GDD


def Anatoxin_Predict(Lag=2, Dis = 150, Length = 2, Turb=100000, Length_t = 2,  GDDmin = 7):
    
    #Combine Toxin Data set with VRdf
    VRtox = pd.merge(VRdf, Tox, how='outer', on='Date' )
    
    #establish empty 
    VRtox['Count'] = np.nan
    
    #convert Na values to 0
    VRtox['Turbidity'] = VRtox['Turbidity'].fillna(0)
    
    #establish empty column for counting days since high discharge
    VRtox['Count 2'] = np.nan
    
    for i in range (14610,len(VRtox),1):
        
        cntii = 0
        test = False
        while test == False:  
            for j in range(i,0,-1):
                
                Hdis = VRtox.at[j,'Discharge (cfs)']
                
                if Hdis <= Dis:
                
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
                
    
    for ii in range (14610,len(VRtox),1):
        
        cntiii = 0
        testii = False
        while testii == False:  
            for jj in range(ii,0,-1):
                
                Turb_size = VRtox.at[jj,'Turbidity']
                
                if Turb_size <= Turb:
                
                    cntiii += 1
                    
                    if jj == 14609:
                        
                        testii = True
                        break
                    
                    else:
                        jj += -1
                        
                else:
                    VRtox.at[ii, 'Count 2'] = cntiii
                    testii = True  
                    break



    VRdf['GDD (Daily)'] = np.nan
    VRdf['GDD (Cumulative)'] =np.nan
    
    for g in range (14610,len(VRdf),1):
        
        AVWT = VRtox.at[g, 'Avg W Temp (C)']
        CNN = VRtox.at[g, 'Count']
        CNt = VRtox.at[g, 'Count 2']
        
        if AVWT <= GDDmin or CNN <= Length or CNt<= Length_t:
            
            VRtox.at[g,'GDD (Daily)'] = 0
            VRtox.at[g,'GDD (Cumulative)'] = 0
            
        else:
            VRtox.at[g,'GDD (Daily)'] = AVWT - GDDmin
            VRtox.at[g,'GDD (Cumulative)'] = (AVWT - GDDmin) + VRtox.at[g-1,'GDD (Cumulative)']
    
    
    #shift column number of days established in definition
    VRtox['Anatoxin-a concentration'] = VRtox['Anatoxin-a concentration'].shift(0-Lag)

   #Plot GDD and Anatoxin-a relationships over time
    A = VRtox['Anatoxin-a concentration']
    Gd = VRtox['GDD (Daily)']
    Gc= VRtox['GDD (Cumulative)']
    D = VRtox['Date']
    
    
    ##Plot GDD and Anatoxin-a Concentrations Overtime
    # create figure and axis objects with subplots()
    fig,ax = plt.subplots()
    # make a plot
    ax.plot(D, Gd, color="green")
    # set x-axis label
    ax.set_xlabel("Date",fontsize=12)
    # set y-axis label
    ax.set_ylabel("Daily GDD (C)",color="green",fontsize=12)
    ax.set_xlim([dt.date(2020, 1, 1), max(D)])
    
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    ax2.plot(D, A,color="red",marker="o")
    ax2.set_ylabel("Anatoxin-a Concentration (ug/L)",color="red",fontsize=12)
    plt.show()
    
    fig2,ax3 = plt.subplots()
    # make a plot
    ax3.plot(D, Gc, color="green")
    # set x-axis label
    ax3.set_xlabel("Date",fontsize=12)
    # set y-axis label
    ax3.set_ylabel("Cumulative GDD (C)",color="green",fontsize=12)
    ax3.set_xlim([dt.date(2020, 1, 1), max(D)])
    
    # twin object for two different y-axis on the sample plot
    ax4=ax3.twinx()
    # make a plot with different y-axis using second axis object
    ax4.plot(D, A,color="red",marker="o")
    ax4.set_ylabel("Anatoxin-a Concentration (ug/L)",color="red",fontsize=12)
    plt.show()
    
    
    #Calculate relationship between GDD and Anatoxin-a 
    Bloom = pd.DataFrame(columns =['Date','GDD (Daily)','GDD (Cumulative)','Anatoxin-a concentration'])
    Bloom = VRtox[['Date','GDD (Daily)','GDD (Cumulative)','Anatoxin-a concentration']]
    Bloom.dropna(subset = ["Anatoxin-a concentration"], inplace=True)
    
    AA = Bloom['Anatoxin-a concentration']
    GGd = Bloom['GDD (Daily)']
    GGc = Bloom['GDD (Cumulative)']
    
    #Build linear relationship between GDD (daily/cumulative) and anatoxin a
    slope4, intercept4, r_value4, p_value4, std_err4 = scipy.stats.linregress(GGd, AA)
    r2 = round((r_value4**2),3)
    
    slope5, intercept5, r_value5, p_value5, std_err5 = scipy.stats.linregress(GGc, AA)
    r2ii = round((r_value5**2),3)
    
    # establish linear regression line
    xx = [0, 18]
    yy = [intercept4,(((18)*slope4)+intercept4)]
    
    mx_GDDc = max(Bloom['GDD (Cumulative)'])
    xii = [0, mx_GDDc]
    yii = [intercept5,((mx_GDDc*slope5)+intercept5)]
    
    
    ## Plot linear regression equations
    plt.figure(2)
    plt.ylabel('Anatoxin-a Concentration (ug/L)')
    plt.xlabel('Daily GDD (C)')
    plt.ylim([-5,700])
    plt.xlim([0,18])
    plt.scatter(GGd, AA, color='blue', marker='o')
    plt.plot(xx, yy, color = 'black')
    plt.annotate('r^2 =', (1,500))
    plt.annotate(r2,(3, 500))
    
    plt.figure(3)
    plt.ylabel('Anatoxin-a Concentration (ug/L)')
    plt.xlabel('Cumulative GDD (C)')
    plt.ylim([-5,700])
    plt.xlim([0, 1600])
    plt.scatter(GGc, AA, color='blue', marker='o')
    plt.plot(xii, yii, color = 'black')
    plt.annotate('r^2 =', (200,500))
    plt.annotate(r2ii,(300, 500))