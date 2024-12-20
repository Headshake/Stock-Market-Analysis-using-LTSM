import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input

def fgsm(input_data, model, true_label, epsilon=0.1, loss=0.0, min_value = 0.0):
    
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
    #while np.any (adversarial_example < min_value):
    #    adversarial_example = tf.clip_by_value(adversarial_example, clip_value_min=min_value, clip_value_max=tf.float32.max)
    print (adversarial_example.numpy())

    return adversarial_example.numpy()

## Read dataset of each stock into dataframe ##
rootpath = os.path.dirname(__file__)
print (rootpath)
folderpath = os.path.join(rootpath, './Stocks')
for file_name in os.listdir(folderpath):
    if file_name.endswith('.csv'):
        df = pd.read_csv(os.path.join(folderpath,file_name),delimiter=',',usecols=['Date','Open','High','Low','Close','Adj Close','Volume'])
        basename, ext = os.path.splitext(file_name)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')  ## Convert 'Date' column to datetime
        #df.set_index('Date', inplace=True)  # Set 'Date' as the index
        #df = df.sort_values(by=['Date'])
    #data_graph_plot(basename)
    print ('file:', file_name)
    data = df.filter(['Close'])

    # Convert the dataframe to a numpy array
    dataset = data.values

    scaler = MinMaxScaler(feature_range=(0,1))
    scale_data = scaler.fit_transform(dataset)

    # Get the number of rows to train the model on
    training_data_len = int(np.ceil(len(dataset)* 0.8))
    print (training_data_len)

    ## Create the training data set ##
    ## Create the scaled training data set ##
    train_data = scale_data[0:int(training_data_len), :]
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
    
    model = load_model(os.path.join(os.path.dirname(__file__), './lstm_model.h5'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    ## Train the model ##
    model.fit(x_train, y_train, batch_size=32, epochs=1,validation_split=0.2)

    ## Create the testing dataset ##
    #test_data = scale_data[training_data_len - 50: , :]

    x_full = scale_data[50:len(dataset),0] 
    x_full = np.concatenate((np.zeros((50)), x_full))
    print (x_full)
    y_full = scale_data[0:len(dataset),0]
    x_full, y_full = np.array(x_full), np.array(y_full)

    ## Reshape the data ##
    x_full = np.reshape(x_full, (x_full.shape[0], 1, 1))
    y_full = np.reshape(y_full, (y_full.shape[0], 1))

    ## Generate adversarial examples for the training data ##
    adversarial_train_data = fgsm(x_full, model, y_full)
    print (adversarial_train_data.shape)

    ## Evaluate the model with adversarial training data
    print(f'Evaluate the model with adversarial training data...')
    train_predictions = model.predict(adversarial_train_data)
    train_rmse = np.sqrt(np.mean((train_predictions - y_full) ** 2)) 
    print (f'Training RMSE with adversarial examples: {train_rmse}')

    # Reshape the adversarial data to match the original shape for inverse transformation
    adversarial_train_data_inv = adversarial_train_data.squeeze(axis=1)
    print (adversarial_train_data_inv.shape)
    print (f'Advesarial train data after reshape: {adversarial_train_data_inv}')
    adversarial_train_data_inv = scaler.inverse_transform(adversarial_train_data_inv)
    print (f'Adversarial train data after scaled: {adversarial_train_data_inv}')
    #adversarial_train_data_inv = [sequence[-1] for sequence in adversarial_train_data_inv]
    #print (f'Adversarial train data 1D: {adversarial_train_data_inv}')

    df_adversarial = pd.DataFrame(adversarial_train_data_inv, columns=['Close'])
    df_adversarial['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

    # Save the modified DataFrame back to the CSV file
    df_adversarial.to_csv(os.path.join(os.path.join(rootpath, './Stocks'), 'Attacked_' + file_name), index=False)
