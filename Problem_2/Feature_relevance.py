import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split #used to split the data into training and learning sets


df = pd.read_csv("OHEcleanedData.csv")
# Assuming `df` is your DataFrame with features and `target` as your target variable
correlation = df.corrwith(df['credit_score']).sort_values(ascending=False)
print(correlation)

#define the inputs
xinp = df.drop(columns = ['credit_score'])
#define the output
yout = df['credit_score']

#data split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    xinp, yout, test_size=0.3, random_state=101) 

# Calculate mutual information
mi_scores = mutual_info_regression(X_train, y_train)
mi_scores = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
print(mi_scores)


