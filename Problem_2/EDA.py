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


# Plot distributions of numerical columns
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.show()
    
# Plot bar charts for catagorical columns
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=col, order=df[col].value_counts().index)
    plt.title(f"Value Counts of {col}")
    plt.xticks(rotation=45)
    plt.show()

# Boxplots for numerical outliers
for col in numerical_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot of {col}")
    plt.show()

#since the data consists of both catagorical and numerical data, the oultliers need to be taken care of separatley

#boxplot shows a lot of outliers. I will use IQR to detect and remove numerical outliers but the winsorize library can also be used
# Calculate IQR
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)  # 25th percentile
    Q3 = df[column].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter the DataFrame
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numerical_cols:
    df = remove_outliers_iqr(df, col)

print("After Removing numerical Outliers")
print(df.describe)

#catagorical outliers. i can either group the outliers into one catagory and remove them all together
#i will try grouping them first then removing them second
# Define a threshold for rare categories (e.g., less than 5% of total rows)
threshold = 0.05 * len(df)

#groups putliers
for col in categorical_cols:
    value_counts = df[col].value_counts()
    rare_categories = value_counts[value_counts < threshold].index
    df[col] = df[col].replace(rare_categories, 'Other')

#removes outliers
#for col in categorical_cols:
#    value_counts = df[col].value_counts()
#    rare_categories = value_counts[value_counts < threshold].index
#    df = df[~df[col].isin(rare_categories)]

#one hot encoding can be used for non-ordinal data, which in this case the data is
encoded_data = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
encoded_data = encoded_data.astype(int)

print("Data cleaning finsihed!")
print(df.describe)
encoded_data.to_csv('OutlierfreeOHE.csv', index=False)

