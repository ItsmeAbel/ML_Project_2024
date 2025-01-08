import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

RawData = pd.read_csv("OutlierfreeOHE.csv") #read the csv file using pandas

#define the inputs
xinp = RawData.drop(columns=['credit_score'])
#define the output
yout = RawData['credit_score']

#data split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    xinp, yout, test_size=0.3, random_state=101) 

# Standardize the data (important for SVR)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.to_numpy().reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.to_numpy().reshape(-1, 1)).ravel()

# Create and train the SVR model
svr = SVR(kernel='rbf', C=1.0)  # Use a linear kernel for multilinear regression
svr.fit(X_train_scaled, y_train_scaled)

# Predict on the test set
y_pred_scaled = svr.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))  # Inverse transform to original scale

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")

#--------plot graph ----------------------
plt.figure(1)
# plotting a scatterplot
plt.scatter(y_test, y_pred, color='red', alpha=0.7)
#sns.scatterplot(x='oldagreement',
              # y='newagreement', data=df)

#plt.plot(X_result, predictions, color='blue', linewidth=2, label='avtalspris')

# Add labels and a legend
plt.xlabel('Target values')
plt.ylabel('Predicted target values')
plt.title('SVR')
# Plot the data as a line graph
# draws the graph
plt.grid(True)
plt.show()
