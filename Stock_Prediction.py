from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates

## Creating the data visualisation ##
def data_graph_plot(fname):
    plt.figure(figsize = (18,9))
    plt.plot(df['Close'])
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price',fontsize=18)
    plt.title(fname+' Stock Closing Price', fontsize=20)
    plt.show()

## Visualise the predicted data ##
def predict_graph_plot(fname, predictedData, prdtData, training_data_len):
    ## Plot the data
    train = predictedData[:training_data_len]
    valid = predictedData[training_data_len:]
    valid.loc[:, 'Predictions'] = prdtData

    plt.figure(figsize=(16,6))
    plt.title(fname + ' Stock Predicted Closing Price')
    plt.xlabel('Date', fontsize = 18)
    plt.ylabel('Close Price', fontsize=18)

    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])

    plt.legend(['Train', 'True', 'Predictions'], loc='lower right')
    plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    #plt.xticks(range(0, df.shape[0], 500), df.index[::500].strftime('%d-%m-%Y'), rotation=45)
    plt.show()

def predict_window(model, last_data, predict_days):
    PredictData = []
    current_data = last_data

    for d in range(predict_days):
        ## prepare the last 200 days of data ## 
        input_data = np.array(current_data[-200:])
        input_data = np.reshape(input_data, (1, input_data.shape[0],1))

        ## make a prediction ##
        predict_week = model.predict(input_data)
        PredictData.append(predict_week[0][0])

        ## append the prediction to the current data ##
        current_data = np.append(current_data, predict_week)
    
    ## Inverse transform the predictions ##
    PredictData = scaler.inverse_transform(np.array(PredictData).reshape(-1,1))
    print (PredictData)
    return PredictData

def plot_week_forecast(fname, df, nextweek_prediction):
    plt.figure(figsize=(16,6))
    plt.plot(df['Date'].iloc[-1], df['Close'].iloc[-1], label="Last day Closing Price")

    last_price = df['Date'].iloc[-1]
    prediction_dates = pd.date_range(last_price, periods=len(nextweek_prediction)+1, inclusive='right')

    plt.plot(prediction_dates, nextweek_prediction, label='Next week predicted Prices', color='red')

    ## Annotate the last actual close price ##
    plt.annotate(f'Last Closing Price: {df["Close"].iloc[-1]:.2f}',
                (df['Date'].iloc[-1], df['Close'].iloc[-1]),
                textcoords='offset points',
                xytext=(0,10),
                ha = 'center',
                color = 'blue')


    ## Annotate the predicted prices ##
    for i in range(len(nextweek_prediction)):
        plt.annotate(f'{nextweek_prediction[i][0]:.2f}',
                    (prediction_dates[i], nextweek_prediction[i][0]),
                    textcoords ="offset points",
                    xytext=(0,10),
                    ha = 'center')

    plt.xlabel('Date')
    plt.ylabel('Predicted Close Price')
    plt.legend()
    #plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
    plt.xticks(rotation=45)
    plt.title(fname + ' Stock Price Predictions')

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())   ## Set major locator to show every week
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))

    plt.show()

## Read dataset of each stock into dataframe ##
rootpath = os.path.dirname(__file__)
print (rootpath)
folderpath = os.path.join(rootpath, './Stocks')
for file_name in os.listdir(folderpath):
    if file_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(folderpath,file_name),delimiter=',',usecols=['Close', 'Date'])
        basename, ext = os.path.splitext(file_name)
        print (file_name)
        df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y", dayfirst=True)  ## Convert 'Date' column to datetime
        #df.set_index('Date', inplace=True)  # Set 'Date' as the index
        #df = df.sort_values(by=['Date'])
        print ('file:', file_name)

        data = df.filter(['Close'])
    
        # Convert the dataframe to a numpy array
        dataset = data.values

    ## Stock Market Analysis + Prediction using LSTM ##
    ## Author: Fares Sayah ##
    ## Reference link: https://www.kaggle.com/code/faressayah/stock-market-analysis-prediction-using-lstm ##
        
        # Get the number of rows to train the model on
        training_data_len = int(np.ceil(len(dataset)* 0.8))

        print (training_data_len)

        scaler = MinMaxScaler(feature_range=(0,1))
        scale_data = scaler.fit_transform(dataset)

        print(scale_data)

        ## Create the training data set ##
        ## Create the scaled training data set ##
        train_data = scale_data[0:int(training_data_len), :]

        ## Split the data into train_data and test_data sets ##
        x_train = []
        y_train = []

        for i in range(50, len(train_data)):
            x_train.append(train_data[i-50:i, 0])
            y_train.append(train_data[i,0])
            if i < 50:   
                print(x_train)
                print(y_train)
                print()

        ## Convert the x_data and y_data to numpy arrays ##
        x_train, y_train = np.array(x_train), np.array(y_train)

        ## Reshape the data
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        ## Build the LSTM model ##
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(64, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))

        ## Compile the model ##
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        ## Train the model ##
        model.fit(x_train, y_train, batch_size=32, epochs=50,validation_split=0.2)

        ## Save the trained model
        model.save(os.path.join(rootpath, 'lstm_model.h5'))

        ## Create the testing dataset ##
        test_data = scale_data[training_data_len - 50: , :]

        ## Create the datasets x_test
        x_test = []
        y_test = dataset[training_data_len: , :]
        for i in range (50, len(test_data)):
            x_test.append(test_data[i-50:i, 0])

        ## Convert the data to a numpy array
        x_test = np.array(x_test)

        ## Reshape the data ##
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

        ## Store the models predicted price values ##
        predict_data = model.predict(x_test)
        predict_data = scaler.inverse_transform(predict_data)

        ## Calculate the root mean squared error (RMSE) ##
        rmse = np.sqrt(np.mean(((predict_data - y_test) ** 2)))

        print(rmse)

        # Calculate Mean Absolute Precentage Error ##
        mape = np.mean(np.abs((y_test - predict_data) / y_test)) * 100
        accuracy = 100 - mape

        print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(mape))
        print('Accuracy: {:.2f}%'.format(accuracy))

        predict_graph_plot(basename,data,predict_data,training_data_len)

        true_data = scale_data[-200:]

        ## predict the next 7 days ##
        nextweek_prediction = predict_window(model, true_data, 7)
        plot_week_forecast(basename, df, nextweek_prediction)