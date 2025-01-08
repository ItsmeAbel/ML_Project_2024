import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load dataset (replace 'your_data.csv' with your file)
data = pd.read_csv('cleaned_merged_heart_dataset.csv')

# Split dataset into features (X) and target (y)
X = data.iloc[:, :-1].values  # Input features (all columns except the last)
y = data.iloc[:, -1].values   # Target column (last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (SVM works better with scaled features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the SVM model
model = SVC(kernel='rbf', random_state=42, C=3.0)  # 'linear', 'rbf', 'poly', etc. for different kernels
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#--------plot graph ----------------------
plt.figure(1)
# plotting a scatterplot
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
#sns.scatterplot(x='oldagreement',
              # y='newagreement', data=df)

#plt.plot(X_result, predictions, color='blue', linewidth=2, label='avtalspris')

# Add labels and a legend
plt.xlabel('Target values')
plt.ylabel('Predicted target values')
plt.title('SVM')
# Plot the data as a line graph
# draws the graph
plt.grid(True)
plt.show()
