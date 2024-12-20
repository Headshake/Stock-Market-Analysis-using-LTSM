import numpy as np
import pandas as pd
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.data.experimental.enable_debug_mode()
from tensorflow.keras.models import load_model
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import ProjectedGradientDescent
from art.utils import load_mnist
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_percentage_error

# PGD model source code: https://github.com/Trusted-AI/adversarial-robustness-toolbox?tab=readme-ov-file

# Load your model
model = load_model(os.path.join(os.path.dirname(__file__), './lstm_model.h5'))
# Compile the model with an Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

#@tf.function
def generate_adversarial_examples(x_test, attack):
    return attack.generate(x=x_test)

loss_object = tf.keras.losses.MeanSquaredError()
classifier = TensorFlowV2Classifier(
    model=model,
    nb_classes=2,  # Adjust this to the number of classes you have
    input_shape=(50, 1),  # Adjust this to your input shape
    loss_object=loss_object,
)

# Set PGD attack parameters
pgd_attack = ProjectedGradientDescent(
    estimator=classifier,
    norm=np.inf,             # L-infinity norm
    eps=0.6,          # Maximum perturbation
    eps_step=0.5,       # Attack step size
    max_iter=20,       # Number of attack iterations
    targeted=False      # Set to True if it's a targeted attack
)


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
    
    cleandata = np.reshape(scale_data, (scale_data.shape[0], 1))

    #adv_data = scale_data.reshape((-1, 50, 1))  # Rshape to the model's input shape

    cleandata = np.array(cleandata)
    print (f'the original test data:{cleandata}')

    # Generate adversarial examples
    adv_data = generate_adversarial_examples(cleandata,pgd_attack)

    ## Create the testing dataset ##
    test_data = adv_data[training_data_len - 50: , :]
    
    ## Create the datasets x_test
    x_test = []
    y_test = dataset[training_data_len: , :]
    for i in range (50, len(test_data)):
        x_test.append(test_data[i-50:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
    
    print ("Adversarial example:", adv_data)

    # Check the difference
    difference = adv_data - cleandata
    #print("Difference:", difference.flatten())

    adv_test = adv_data[training_data_len: , :]

    adv_data = scaler.inverse_transform(adv_data)
    print (adv_data)

    # Evaluate the model on adversarial examples
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    print (f'Predictions: {predictions}')
    mape = mean_absolute_percentage_error(y_test, predictions)
    accuracy = 1 - mape
    print(f'MAPE on adversarial examples: {mape * 100:.2f}%')   
    print(f'Accuracy on adversarial examples: {accuracy * 100:.2f}%')   

    # Inverse transform the adversarial examples to the original scale
    x_test_adv_rescaled = adv_data.reshape(-1, 1)

    # Flatten x_test_adv_rescaled if it's a multi-dimensional array and needs flattening
    x_test_adv_rescaled_flattened = x_test_adv_rescaled.flatten()
    print (x_test_adv_rescaled_flattened)

    # Create a new DataFrame with 'Date' from the original df and 'Close' from the adversarial data
    adversarial_df = pd.DataFrame({
        'Close': x_test_adv_rescaled_flattened,
        'Date': df['Date']
    })

    # Save the new DataFrame to a CSV file
    adversarial_df.to_csv(os.path.join(os.path.join(rootpath, './Stocks'), 'Attacked_' + file_name), index=False)
    print(f'Adversarial data saved to Attacked_{file_name}')