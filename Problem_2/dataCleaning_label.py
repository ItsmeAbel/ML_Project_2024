from sklearn.preprocessing import LabelEncoder #used for encoding catagorcal string data
import pandas as pd

RowData = pd.read_csv('loan_data.csv')
#label encoder can be used for ordinal data, but if it can increase model bias if used on non-ordinal data since it applies order in the data that doesn't exist
encoder = LabelEncoder()

RowData['person_gender']= encoder.fit_transform(RowData['person_gender'])
RowData['person_education']= encoder.fit_transform(RowData['person_education'])
RowData['person_home_ownership']= encoder.fit_transform(RowData['person_home_ownership'])
RowData['loan_intent']= encoder.fit_transform(RowData['loan_intent'])
RowData['previous_loan_defaults_on_file']= encoder.fit_transform(RowData['previous_loan_defaults_on_file'])


print(RowData)
RowData.to_csv('LabelcleanedData.csv', index=False)

