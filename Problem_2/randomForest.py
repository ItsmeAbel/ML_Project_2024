# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Step 1: Load the CSV File
file_path = "LabelcleanedData.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Step 2: Explore the Data
print("Data Preview:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

# Step 3: Separate Features and Target
# Assume the target column is named 'target'. Replace with your actual target column name.
X = data.drop(columns=['credit_score'])
y = data['credit_score']

# Step 4: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

# Step 8: Analyze Feature Importance (Optional)
feature_importances = model.feature_importances_
features = X.columns

print("\nFeature Importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")
