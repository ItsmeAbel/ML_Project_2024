from sklearn.preprocessing import LabelEncoder #used for encoding catagorcal string data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('loan_data.csv')

# Inspect the dataset
print(df.head())        # First 5 rows
print(df.info())        # Data types and missing values
print(df.describe())    # Statistical summary for numerical features. shows information such as min, mean and max values for each feature

# Check for missing values. shows 0 if none are missing
print(df.isnull().sum())

# Check for duplicated rows
print(f"Duplicated rows: {df.duplicated().sum()}")

# Select categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
# Select numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

#one hot encoding can be used for non ordinal data, which in this case there are
encoded_data = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
encoded_data = encoded_data.astype(int)

numerical_cols_OHE = encoded_data.select_dtypes(include=['float64', 'int64']).columns
# Plot distributions
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()
    
# Plot bar charts
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Value Counts of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Boxplots for outlier detection
for col in numerical_cols_OHE:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=encoded_data, x=col)
    plt.title(f"Boxplot of {col}")
    plt.show()



#boxplot shows a lot of outliers. I will use IQR to detect and remove these outliers
# Calculate IQR
Q1 = encoded_data.quantile(0.25)
Q3 = encoded_data.quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers
outlier_mask = ~((encoded_data < (Q1 - 1.5 * IQR)) | (encoded_data > (Q3 + 1.5 * IQR))).any(axis=1)
df_cleaned = df[outlier_mask]



print(df_cleaned)
df_cleaned.to_csv('OutlierfreeOHE.csv', index=False)

