import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Activation, Softmax
from keras.activations import elu, softmax, tanh, sigmoid, relu, softplus
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


# Load the CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocessing: handle missing values, split data, and scale features
def preprocess_data(data, target_column):
    # Drop rows with missing target values
    data = data.dropna(subset=[target_column])
    
    # Fill missing feature values with the mean
    data.fillna(data.mean(), inplace=True)
    

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_test = scaler.transform(X_test)

    # Scale the target variable
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler.fit_transform(y_test.reshape(-1, 1))

    return X_train, X_test, y_train_scaled, y_test_scaled, scaler

# Build the neural network model
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, activation=act_func, input_dim=input_dim))
    model.add(Dense(30, activation=act_func))
    model.add(Dense(65, activation=act_func))
    model.add(Dense(90, activation=act_func))
    model.add(Dense(120, activation=act_func))
    model.add(Dense(90, activation=act_func))
    #model.add(Dropout(0.2))
    model.add(Dense(65, activation=act_func))
    #model.add(Dropout(0.2))
    model.add(Dense(30, activation=act_func))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# Train the model
def train_model(model, X_train, y_train):
    #early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=10, 
                        batch_size=100, 
                        verbose=1)
    return history

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    return predictions

# Main function
def main(csv_file, target_column):
    data = load_data(csv_file)
    X_train, X_test, y_train_scaled, y_test_scaled, scaler = preprocess_data(data, target_column)
    
    model = build_model(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train_scaled)
    
    print("Evaluating model on test set:")
    predictions = evaluate_model(model, X_test, y_test_scaled)

    pred_scaled= scaler.inverse_transform(model.predict(predictions))

    return model, scaler, pred_scaled

# Usage example
# Replace 'data.csv' with the path to your dataset and 'target' with your target column name.
csv_file = 'OHEcleanedData.csv'
target_column = 'credit_score'
act_func = "linear"
scaler = MinMaxScaler(feature_range=(0, 1))
model, scaler, predictions = main(csv_file, target_column)
