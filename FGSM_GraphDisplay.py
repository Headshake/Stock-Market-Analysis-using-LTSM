import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input

def fgsm(input_data, model, true_label, epsilon, loss=0.0, min_value = 0.0):
    
    """
    Generate adversarial examples using FGSM.
    :param input_data: Input data to be perturbed
    :param model: Trained model
    :param epsilon: Perturbation size
    """
    input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
    loss = tf.convert_to_tensor(loss, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(input_data)
        prediction = model(input_data)
        loss = tf.keras.losses.MeanSquaredError()(true_label, prediction)
        #penalty = tf.reduce_sum(tf.square(tf.minimum(loss, 0)))
    
    gradient = tape.gradient(loss, input_data)
    adversarial_example = input_data + epsilon * tf.sign(gradient)
    while np.any (adversarial_example < min_value):
        adversarial_example = tf.clip_by_value(adversarial_example, clip_value_min=min_value, clip_value_max=tf.float32.max)

    print (adversarial_example.numpy())

    return adversarial_example.numpy()

## Visualise the predicted data ##
def advexmp_graph_plot(fname, CleanData, Clean_PredictedData, training_data_len, AdvData, Adv_PredictData, dates):
    ## Plot the original clean predicted data
    clean_valid = CleanData[training_data_len:]
    clean_valid.loc[:, 'Predictions'] = Clean_PredictedData
    
    ## Plot the adversarial attacked data forecast price
    adv_valid = AdvData[training_data_len:]
    adv_valid.loc[:, 'Adv Predictions'] = Adv_PredictData

    plt.figure(figsize=(18,6))
    plt.title(fname + ' Stock Predicted Closing Price')
    plt.xlabel('Year', fontsize = 18)
    plt.ylabel('Close Price', fontsize=18)

    plt.plot(dates, clean_valid[['Close','Predictions']])
    plt.plot(dates, adv_valid['Adv Predictions'], color='red', linestyle='dotted')

    plt.legend(['Clean True Value', 'Clean Predictions', 'Adv Predictions'], loc='upper left')
    plt.margins(x=0.0005, y=0.01)
    plt.xlim([dates.iloc[0], dates.iloc[-1]])

    plt.show()

def plot_epsilon_graph (fname, actual_data, adv_data_01, adv_data_04, adv_data_06, dates):
    plt.figure(figsize=(18,6))
    plt.plot(dates, actual_data, label='Actual Data', color='blue')
    plt.plot(dates, adv_data_01, label='Adversarial Input (epsilion=0.1)', color='red', linestyle='--')
    plt.plot(dates, adv_data_04, label='Adversarial Input (epsilion=0.4)', color='orange', linestyle='-.')
    plt.plot(dates, adv_data_06, label='Adversarial Input (epsilion=0.6)', color='green', linestyle='dotted')

    plt.title(f'FGSM-Generated Adversarial Examples for {fname} with Different Epsilion Values')
    plt.xlabel('Year')
    plt.ylabel('Closing Price ($)')
    plt.legend(['True', 'Adv Data(epsilion=0.1)', 'Adv Data(epsilion=0.4)', 'Adv Data(epsilion=0.6)'], loc='upper left')
    plt.show()

def create_training_data (data, datalength):
    ## Create the training data set ##
    ## Create the scaled training data set ##
    train_data = data[0:int(datalength), :]
    print (train_data)
    
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
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    return x_train, y_train

def create_test_data (testdata, dataset):
    ## Create the datasets x_test
    x_test = []
    y_test = dataset[training_data_len: , :]
    for i in range (50, len(testdata)):
        x_test.append(testdata[i-50:i, 0])

    x_test = np.array(x_test)

    ## Reshape the data ##
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    return x_test, y_test

def cal_error (true_label, predict_data):
    # Calculate Mean Absolute Precentage Error ##
    mape = np.mean(np.clip((np.abs((true_label - predict_data) / true_label)) * 100, 0,100))
    accuracy = 100 - mape
    return mape, accuracy

## Read dataset of each stock into dataframe ##
rootpath = os.path.dirname(__file__)
print (rootpath)
folderpath = os.path.join(rootpath, './Stocks')
# Get all the filenames in the directory
all_files = os.listdir(folderpath)
# Filter files that have 'Attacked' in their name
attacked_files = [f for f in all_files if 'Attacked' in f]
# Filter files that do not have 'Attacked' but share the same base filename
non_attacked_files = [f for f in all_files if 'Attacked' not in f]

# Initialize lists to store matched pairs of files
paired_files = []
# Find corresponding non-attacked files for each attacked file
for attacked_file in attacked_files:
    base_name = attacked_file.replace('Attacked_', '').strip()
    matching_files = [f for f in non_attacked_files if base_name in f]
    if matching_files:
        paired_files.append((attacked_file, matching_files[0]))

for attacked_file, original_file in paired_files:
    scaler = MinMaxScaler(feature_range=(0,1))

    df_adv = pd.read_csv(os.path.join(folderpath,attacked_file),delimiter=',',usecols=['Close', 'Date'])
    basename, ext = os.path.splitext(attacked_file)
    adv_data = df_adv.filter(['Close'])
    dataset_adv = adv_data.values
    scale_adv_data = scaler.fit_transform(dataset_adv)
    # Get the number of rows to train the model on
    adv_datalen = int(np.ceil(len(dataset_adv)* 0.8))
    print (adv_datalen)
    
    df = pd.read_csv(os.path.join(folderpath,original_file),delimiter=',',usecols=['Date','Open','High','Low','Close','Adj Close','Volume'])
    basename, ext = os.path.splitext(original_file)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)  ## Convert 'Date' column to datetime
    print ('file:', basename)
    data = df.filter(['Close'])
    # Convert the dataframe to a numpy array
    dataset = data.values
    scale_data = scaler.fit_transform(dataset)

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset)* 0.8))
    print (training_data_len)

    clean_xtrain, clean_ytrain = create_training_data(scale_data, training_data_len)
    adv_xtrain, adv_ytrain = create_training_data(scale_adv_data, adv_datalen)

    ## Load the trained model
    model = load_model(os.path.join(os.path.dirname(__file__), './lstm_model.h5'))
    model.compile(optimizer='adam', loss='mean_squared_error')

    ## Train the model ##
    #model.fit(clean_xtrain, clean_ytrain, batch_size=32, epochs=10,validation_split=0.2)
    ## Create the testing dataset ##
    test_data = scale_data[training_data_len - 50: , :]
    clean_xtest, clean_ytest = create_test_data (test_data, dataset)
    
    ## Store the models predicted price values ##
    clean_predict_data = model.predict(clean_xtest)
    clean_predict_data = scaler.inverse_transform(clean_predict_data)

    cleanMAPE, cleanAccuracy = cal_error(clean_ytest, clean_predict_data)
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(cleanMAPE))
    print('Accuracy: {:.2f}%'.format(cleanAccuracy))


    #model.fit(adv_xtrain, adv_ytrain, batch_size=32, epochs=10,validation_split=0.2)
    adv_testdata = scale_adv_data[adv_datalen - 50: , :]
    adv_xtest, adv_ytest = create_test_data (adv_testdata, dataset_adv)
    
    adv_predict_data = model.predict(adv_xtest)
    adv_predict_data = scaler.inverse_transform(adv_predict_data)
    advMAPE, advAccuracy = cal_error(adv_ytest, adv_predict_data)
    print('Mean Absolute Percentage Error (MAPE): {:.2f}%'.format(advMAPE))
    print('Accuracy: {:.2f}%'.format(advAccuracy))

    advexmp_graph_plot(basename, data, clean_predict_data, training_data_len, adv_data, adv_predict_data, df['Date'][training_data_len:])

    x_full = scale_data[50:len(dataset),0] 
    x_full = np.concatenate((np.zeros((50)), x_full))
    print (x_full)
    y_full = scale_data[0:len(dataset),0]
    x_full, y_full = np.array(x_full), np.array(y_full)

    ## Reshape the data ##
    x_full = np.reshape(x_full, (x_full.shape[0], 1, 1))
    y_full = np.reshape(y_full, (y_full.shape[0], 1))

    ## Generate adversarial examples for the data ##
    adversarial_data01 = fgsm(x_full, model, y_full, epsilon=0.1)
    adversarial_data04 = fgsm(x_full, model, y_full, epsilon=0.4)
    adversarial_data06 = fgsm(x_full, model, y_full, epsilon=0.6)

    print(adversarial_data06)

    ## Evaluate the model with adversarial training data
    print(f'Evaluate the model with adversarial data...')

    # Reshape the adversarial data to match the original shape for inverse transformation
    adversarial_data01 = adversarial_data01.squeeze(axis=1)
    print (adversarial_data01.shape)
    print (f'Advesarial train data after reshape: {adversarial_data01}')
    adversarial_data01 = scaler.inverse_transform(adversarial_data01)
    print (f'Adversarial train data after scaled: {adversarial_data01}')

    adversarial_data04 = adversarial_data04.squeeze(axis=1)
    print (adversarial_data04.shape)
    print (f'Advesarial train data after reshape: {adversarial_data04}')
    adversarial_data04 = scaler.inverse_transform(adversarial_data04)
    print (f'Adversarial train data after scaled: {adversarial_data04}')
    #adversarial_train_data_inv = [sequence[-1] for sequence in adversarial_train_data_inv]
    #print (f'Adversarial train data 1D: {adversarial_train_data_inv}')

    adversarial_data06 = adversarial_data06.squeeze(axis=1)
    print (adversarial_data06.shape)
    print (f'Advesarial train data after reshape: {adversarial_data06}')
    adversarial_data06 = scaler.inverse_transform(adversarial_data06)
    print (f'Adversarial train data after scaled: {adversarial_data06}')

    plot_epsilon_graph(basename, data, adversarial_data01, adversarial_data04, adversarial_data06, df['Date'])