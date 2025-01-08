# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

# Step 1: Load the CSV File
file_path = "OutlierfreeOHE.csv"  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

#Explore the Data
print("Data Preview:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

#Separate Features and Target
X = data[['person_age', 'person_emp_exp', 'cb_person_cred_hist_length', 'person_education_Bachelor','person_education_Master',
       'previous_loan_defaults_on_file_No','previous_loan_defaults_on_file_Yes','person_education_High School','person_education_Other']]
y = data['credit_score']

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

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


# Analyze Feature Importance
feature_importances = model.feature_importances_
features = X.columns
# Get feature importances
feature_importance = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_importance)

print("\nFeature Importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")
