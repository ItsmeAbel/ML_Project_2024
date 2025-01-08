import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU, Activation, Softmax, Normalization
from keras.activations import elu, softmax, tanh, sigmoid, relu, softplus
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from joblib import dump

def save_model():
    #dumb the trained model into a joblib file
    dump(model, "NNModel.joblib")


normalizer = Normalization()

# Load the CSV data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocessing: handle missing values, split data, and scale features
def preprocess_data(data, target_column):
    
    # Separate features and target
    X = data[['person_age', 'person_emp_exp', 'cb_person_cred_hist_length', 'person_education_Bachelor','person_education_Master',
       'previous_loan_defaults_on_file_No','previous_loan_defaults_on_file_Yes','person_education_High School','person_education_Other']]
    y = data[target_column]
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

# Build the neural network model
def build_model(input_dim):
    model = Sequential()
    #normalizer
    model.add(Dense(80, activation=act_func, input_dim=input_dim))
    model.add(Dense(95, activation=act_func))
    model.add(Dense(120, activation=act_func))
    model.add(Dense(190, activation=act_func))
    model.add(Dense(120, activation=act_func))
    model.add(Dropout(0.2)) #used for reguralization. drops 20% of the neurons to avoid overfitting
    model.add(Dense(95, activation=act_func))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))  # Output layer for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    return model

# Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
def train_model(model, X_train, y_train):
    #early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, 
                        validation_split=0.2, 
                        epochs=100, 
                        batch_size=100,
                        callbacks=[early_stopping],
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

    print(f"Actual Credit Scores: {y_test[:5]}")
    print(f"Predicted Credit Scores: {predictions[:5]}")
    return predictions

# Main function
def main(csv_file, target_column):
    data = load_data(csv_file)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, target_column)
    
    model = build_model(input_dim=X_train.shape[1])
    train_model(model, X_train, y_train)
    
    print("Evaluating model on test set:")
    predictions = evaluate_model(model, X_test, y_test)

    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()
    return model, scaler, predictions


csv_file = 'OutlierfreeOHE.csv'
target_column = 'credit_score'
act_func = "relu"
model, scaler, predictions = main(csv_file, target_column)
#save the model
save_model()
