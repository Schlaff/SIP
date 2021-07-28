# -*- coding: utf-8 -*-
"""
Created on Tue May 25 08:30:57 2021

@author: brian
"""

import pandas as pd
import glob


#Add the min temperature data
pathI = r'C:\Users\brian\OneDrive\Desktop\SIP\Data\Virgin_River\Water_Temp' 
WaterTempFiles = glob.glob(pathI + "\*.csv")
WaterT = []

for x in WaterTempFiles:
    wtfdf = pd.read_csv(x, index_col=None, header=0)
    WaterT.append(wtfdf)

WTF = pd.concat(WaterT, axis=0, ignore_index=True)

#WTF.to_csv('WaterTempAllYears.csv')
WTF = WTF[['Date','Temp']]
WTF['Date'] = pd.to_datetime(WTF['Date']).dt.tz_localize(None)


#Create an empty dataframe
column_names = ["Date","Max Temp (C)","Min Temp (C)","Avg Temp (C)"]

WT = pd.DataFrame(columns = column_names)

#Group by Date and find Max, Min and Average Temperature
WT['Max Temp (C)'] = WTF.groupby(['Date'], as_index=True)['Temp'].max()
WT['Min Temp (C)'] = WTF.groupby(['Date'], as_index=True)['Temp'].min()
WT['Avg Temp (C)'] = WTF.groupby(['Date'], as_index=True)['Temp'].mean()

WT.to_csv('WT_DNR.csv')