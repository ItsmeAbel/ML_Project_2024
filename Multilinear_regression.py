import pandas as pd
from sklearn.model_selection import train_test_split #used to split the data into training and learning sets
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #used to evaluate the model/test the result
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


RawData = pd.read_csv("cleaned_merged_heart_dataset.csv") #read the csv file using pandas

#define the inputs
xinp = RawData[['age','sex', 'cp','trestbps','chol','fbs','restecg','thalachh','exang','oldpeak','slope','ca']]
#define the output
yout = RawData['target']

#data split into training and testing sets
X_train_Lin, X_test_Lin, y_train_Lin, y_test_Lin = train_test_split(
    xinp, yout, test_size=0.3, random_state=101) 

#model
linregmodel = LinearRegression()

#Fit/taind
linregmodel.fit(X_train_Lin,y_train_Lin)

#Predict/test
result = linregmodel.predict(X_test_Lin)

mse_lin = mean_squared_error(y_test_Lin, result)
r2_lin = r2_score(y_test_Lin, result)
mae= mean_absolute_error(y_test_Lin, result)

#------------print stats ---------------------

print(RawData)
print(xinp)
print(yout)

#Print accuracy measurments
print(
  'mean_squared_error : ', mse_lin)
print(
  'mean_absolute_error : ', mae)
print("R2: ", r2_lin)

#--------plot graph ----------------------
plt.figure(1)
# plotting a scatterplot
plt.scatter(y_test_Lin, result, color='blue', alpha=0.7)
#sns.scatterplot(x='oldagreement',
              # y='newagreement', data=df)

#plt.plot(X_result, predictions, color='blue', linewidth=2, label='avtalspris')

# Add labels and a legend
plt.xlabel('Target values')
plt.ylabel('Predicted target values')
plt.title('Multilinear regression')
# Plot the data as a line graph
# draws the graph
plt.grid(True)
plt.show()
