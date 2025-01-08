import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# loads the csv data
data = pd.read_csv('cleaned_merged_heart_dataset.csv')

# Split into features (X) and target (y)
X = data.iloc[:, :-1].values  # All rows, all columns except the last one
y = data.iloc[:, -1].values   # Target column, which is the last one

# Train-test split (60% training, 40% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Create and train the logistic regression model
model = LogisticRegression(solver= 'lbfgs', max_iter=500, C=0.1)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model and print
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#--------plot graph ----------------------
plt.figure(1)
# plotting a scatterplot
plt.scatter(y_test, y_pred, color='yellow', alpha=0.7)

# Add labels and a legend
plt.xlabel('Target values')
plt.ylabel('Predicted target values')
plt.title('Logistic regression')
# Plot the data as a line graph
# draws the graph
plt.grid(True)
plt.show()
