#1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
data = pd.read_csv('temperature_data.csv')

# Convert 'Date' column to datetime type
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by date (if not already sorted)
data.sort_values('Date', inplace=True)

# Normalize the 'Temperature' column
scaler = MinMaxScaler()
data['Temperature'] = scaler.fit_transform(data['Temperature'].values.reshape(-1, 1))

# Split the data into training and test sets
train_size = int(len(data) * 0.8)  # 80% for training
train_data, test_data = data[:train_size], data[train_size:]

# Optionally, you can reset index for both train_data and test_data
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# Optionally, you can save the preprocessed data to new CSV files
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)




#2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Normalize the data
scaler = MinMaxScaler()
train_data['Temperature'] = scaler.fit_transform(train_data['Temperature'].values.reshape(-1, 1))
test_data['Temperature'] = scaler.transform(test_data['Temperature'].values.reshape(-1, 1))

# Define function to create sequences for input data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Define hyperparameters
sequence_length = 10  # Length of input sequences
n_features = 1  # Number of features (temperature)
n_units = 50  # Number of LSTM units in each layer
dropout_rate = 0.2  # Dropout rate

# Create sequences for training and test data
X_train, y_train = create_sequences(train_data['Temperature'], sequence_length)
X_test, y_test = create_sequences(test_data['Temperature'], sequence_length)

# Define the model
model = Sequential([
    LSTM(n_units, input_shape=(sequence_length, n_features), return_sequences=True),
    Dropout(dropout_rate),
    LSTM(n_units),
    Dropout(dropout_rate),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)

# Optionally, make predictions
predictions = model.predict(X_test)



#3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load preprocessed data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Normalize the data
scaler = MinMaxScaler()
train_data['Temperature'] = scaler.fit_transform(train_data['Temperature'].values.reshape(-1, 1))
test_data['Temperature'] = scaler.transform(test_data['Temperature'].values.reshape(-1, 1))

# Define function to create sequences for input data
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length])
    return np.array(X), np.array(y)

# Define hyperparameters
sequence_length = 10  # Length of input sequences
n_features = 1  # Number of features (temperature)
n_units = 50  # Number of LSTM units in each layer
dropout_rate = 0.2  # Dropout rate

# Create sequences for training and test data
X_train, y_train = create_sequences(train_data['Temperature'], sequence_length)
X_test, y_test = create_sequences(test_data['Temperature'], sequence_length)

# Define the model
model = Sequential([
    LSTM(n_units, input_shape=(sequence_length, n_features), return_sequences=True),
    Dropout(dropout_rate),
    LSTM(n_units),
    Dropout(dropout_rate),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)




#4
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform the predictions and ground truth
predictions = scaler.inverse_transform(predictions)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate evaluation metrics
mae = mean_absolute_error(y_test_inv, predictions)
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))

print("Test Loss (MSE):", test_loss)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

# Visualize predictions
plt.figure(figsize=(12, 6))
plt.plot(y_test_inv, label='Ground Truth', color='blue')
plt.plot(predictions, label='Predictions', color='red')
plt.title('Temperature Forecasting - LSTM Model')
plt.xlabel('Time Steps')
plt.ylabel('Temperature')
plt.legend()
plt.show()



#5
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# Define a function to create the LSTM model
def create_lstm_model(n_units=50, dropout_rate=0.2):
    model = Sequential([
        LSTM(n_units, input_shape=(sequence_length, n_features), return_sequences=True),
        Dropout(dropout_rate),
        LSTM(n_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Wrap the Keras model so it can be used by scikit-learn
keras_model = KerasRegressor(build_fn=create_lstm_model)

# Define hyperparameters to search
param_grid = {
    'n_units': [50, 100, 150],  # Number of LSTM units
    'dropout_rate': [0.2, 0.3, 0.4],  # Dropout rate
    'batch_size': [32, 64, 128],  # Batch size
    'epochs': [50, 100, 150],  # Number of epochs
}

# Perform random search
random_search = RandomizedSearchCV(estimator=keras_model, param_distributions=param_grid,
                                   n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best Parameters:", random_search.best_params_)




#Challenges Encountered During Model Training and Optimization:
#Overfitting: LSTM models can easily overfit, especially when dealing with small datasets or complex architectures. Regularization techniques like dropout and early stopping are crucial to mitigate this issue.
#Hyperparameter Tuning: Finding the optimal set of hyperparameters can be challenging and computationally expensive, especially with a large search space. Techniques like grid search or random search can help, but they still require significant computational resources.#6



#How did you decide on the number of LSTM layers and units?
#Decision on the Number of LSTM Layers and Units:
#The number of LSTM layers and units depends on the complexity of the dataset and the problem at hand. Adding more layers and units can increase the model's capacity to capture complex patterns, but it also increases the risk of overfitting. It's essential to strike a balance between model complexity and generalization performance through experimentation and validation.



#Preprocessing Steps on the Time Series Data:
#Preprocessing steps typically include:
#Normalization: Scaling the data to a range (e.g., [0, 1]) to stabilize training and ensure all features contribute equally.
#Sequence Creation: Creating input-output sequences from the time series data, which involves determining the sequence length and splitting the data accordingly.
#Train-Test Split: Dividing the data into training and test sets to evaluate the model's performance on unseen data.


#Purpose of Dropout Layers in LSTM Networks and Overfitting Prevention:
#Dropout layers are used in LSTM networks to prevent overfitting by randomly setting a fraction of input units to zero during training. This helps in creating a more robust model that generalizes better to unseen data.
#Dropout introduces noise during training, which acts as a form of regularization, forcing the network to learn redundant representations. As a result, the network becomes less sensitive to specific patterns in the training data and exhibits better generalization performance.
#Dropout is particularly effective in deep networks with many parameters, such as LSTM networks, where overfitting is a common concern. By randomly dropping units during training, dropout prevents co-adaptation of neurons and encourages the network to learn more robust features.


#Analyze the model's ability to capture long-term dependencies and make
#accurate predictions.
#Long-Term Dependencies: LSTM networks are designed to capture long-term dependencies in sequential data. The model's architecture, including the number of LSTM layers and units, as well as the sequence length, plays a crucial role in its ability to capture these dependencies. By using multiple LSTM layers and a sufficient number of units, the model can learn complex patterns over longer sequences of input data.

#Prediction Accuracy: The accuracy of the model's predictions can be assessed using evaluation metrics such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE). A low value of these metrics indicates that the model's predictions are close to the ground truth, suggesting that it can accurately forecast future values based on past observations.
#Visual Inspection: Visualizing the model's predictions against the ground truth can provide additional insights into its performance. By plotting the actual and predicted values over time, we can observe how well the model captures the underlying trends and patterns in the data.


#Feature Engineering: Incorporating additional relevant features, such as weather conditions or seasonal patterns, could improve the model's ability to capture complex relationships in the data.

#Ensemble Methods: Combining predictions from multiple LSTM models or different types of models (e.g., LSTM, CNN, or traditional time series models) through ensemble methods like averaging or stacking could lead to better overall performance.

#Model Architecture: Experimenting with different architectures, such as adding attention mechanisms or incorporating exogenous variables, could enhance the model's ability to capture dependencies and make accurate predictions.

#Hyperparameter Tuning: Fine-tuning hyperparameters such as learning rate, batch size, and dropout rate through techniques like grid search or random search could further optimize the model's performance.

#Transfer Learning: Pre-training the LSTM model on a related dataset or using pre-trained embeddings could help bootstrap learning and improve performance, especially when dealing with limited training data.