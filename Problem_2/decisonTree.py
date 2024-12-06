import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pdb

# Step 1: Load the dataset from a CSV file
file_path = 'labelcleanedData.csv'  # Replace with your CSV file path
RawData = pd.read_csv(file_path)

# Step 2: Inspect the dataset (optional)
print(RawData.head())  # View the first few rows
print(RawData.info())  # Check data types and missing values

# Step 3: Define features (X) and target (y)
# Replace 'feature1', 'feature2', etc., and 'target' with your actual column names
X = RawData.drop(columns=['credit_score']) # Select features
y = RawData['credit_score'] #target

#pdb.set_trace() #breakpoint
# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Step 5: Create and train the Decision Tree model
model = DecisionTreeRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
