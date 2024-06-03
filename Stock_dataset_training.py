from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6

def data_graph(fname):
    plt.figure(figsize = (18,9))
    plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Mid Price',fontsize=18)
    plt.title(fname, fontsize=20)
    plt.show()


rootpath = os.path.dirname(__file__)
print (rootpath)
folderpath = os.path.join(rootpath, './Stocks')
for file_name in os.listdir(folderpath):
    if file_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(folderpath,file_name),delimiter=',',usecols=['Date','Open','High','Low','Close','Adj Close','Volume'])
        basename, ext = os.path.splitext(file_name)
        df = df.sort_values('Date')
        df.head()
        data_graph(basename)
    print ('file:', file_name)
#df = pd.read_csv(os.path.join('Stocks','AAPL.csv'),delimiter=',',usecols=['Date','Open','High','Low','Close','Adj Close','Volume'])
print('Loaded data from the YFinance')

