# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 12:22:50 2021

@author: andre
"""

#This program downloads the data for Google stocks in the past 5 years
#Then, it calculates the rolling standard deviation
#Lastly, it plots the stock price data and the results


#import libraries
import numpy as np
import pandas as pd 
import pandas_datareader as web

#we retrieve data from Google itself
goog= web.DataReader('GOOG',data_source='yahoo', 
                    start='3/14/2009', end='4/14/2014')
goog.tail()

#Compute volatility=rolling standard deviation of log returns 
#First, get log of data
goog['Log_Ret']=np.log(goog['Close']/goog['Close'].shift(1))
goog['Volatility']=pd.Rolling.std(goog['Log_Ret'],window=252)*np.sqrt(252)






