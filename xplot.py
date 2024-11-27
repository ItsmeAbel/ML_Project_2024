import pandas as pd
import matplotlib.pyplot as plt

# Load your dataset (replace 'your_data.csv' with your file name)
data = pd.read_csv('cleaned_merged_heart_dataset.csv')

# Assuming your inputs are all columns except the last one
X = data.iloc[:, :-1]  # Select all input features

# Example: Scatter plot between the first two input features
plt.figure(figsize=(8, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], alpha=0.7, edgecolor='k')
plt.title("Scatter Plot of First Two Input Features")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
