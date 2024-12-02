from sklearn.preprocessing import LabelEncoder #used for encoding catagorcal string data
import pandas as pd

RowData = pd.read_csv('loan_data.csv')
#label encoder can be used for ordinal data, but if it can increase model bias if used on non-ordinal data since it applies order in the data that doesn't exist
#encoder = LabelEncoder()

#RowData['person_gender']= encoder.fit_transform(RowData['person_gender'])
#RowData['person_education']= encoder.fit_transform(RowData['person_education'])
#RowData['person_income']= encoder.fit_transform(RowData['person_income'])
#RowData['person_emp_exp']= encoder.fit_transform(RowData['person_emp_exp'])
#RowData['previous_loan_defaults_on_file']= encoder.fit_transform(RowData['previous_loan_defaults_on_file'])

#one hot encoding can be used for non ordinal data, which in this case there are
categorical_columns = RowData.select_dtypes(include=['object', 'category']).columns
#encoded_data = pd.get_dummies(RowData, columns=['person_gender', 'person_emp_exp', 'person_income', 'person_education'], drop_first=False)
encoded_data = pd.get_dummies(RowData, columns=categorical_columns, drop_first=False)
encoded_data = encoded_data.astype(int)

print(encoded_data)
encoded_data.to_csv('OHEcleanedData.csv', index=False)

