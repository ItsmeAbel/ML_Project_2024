# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Step 1: Load the CSV File
file_path = "OHEcleanedData.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

#Explore the Data
print("Data Preview:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

#Separate Features and Target
X = data.drop(columns=['credit_score'])
y = data['credit_score']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
selector = RFE(model, n_features_to_select=10)
selector.fit(X_train, y_train)

# Check selected features
selected_features = X_train.columns[selector.support_]
print("Selected features:", selected_features)

# Make Predictions
y_pred = selector.predict(X_test)

#Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae= mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared (R2): {r2:.2f}")
