To see step by step progress:
https://github.com/ItsmeAbel/ML_Project_2024/tree/main/Problem_1

Prerequistes: Installed Anaconda for the usage of environments
        - Installed all the necessary packages suchas matplotlib, scikit-learn, pandas, numpy,tensorflow, keras etc.

        Thought process in solving the assignment
Got the csv data
The data has multiple inputs and one output, making multilinear regression a good start. Also it is going to be a supervised model
So far, i don't see a need for data cleaning so i will start with it as it is. All the data seems to be numerical and non-string or float.
I will devide the data into training and testing set. 70,30 percent respectively
using anaconda to create a virutal environment for it. The env has numpy,pandas,scikit-learn and matplotlib so far.
implemented a simple multilinear regression model
implemented a simple svr model as well
output graph is similar for both
model need to be finetuned and properly tested
did some finetuning
result wasn't optimal. MSE = 0.15, MAE = 0.35. the closer it is to 1 the better, meaning this is very bad.
added graph using matplotlib
i have analayzed the output graph and am convinced this might be a classification problem instead since the output is either 1 0r 0, yes or no
going to try implementing a binary classification solution
will implement logistic regression for classification. 
logistic regression is clearly a better fitting solution. accuracy-0.73
default itteration is 100, going to increase that 500. same accuracy-0.73
going to use scalar to scale the data. accuracy decrease to 0.72
removed scaler.
added a low reguralization incase there is an overfitting. C=0.1. accuracy 0.73
tried sag and saga solvers. .72 and .71 accuracy respectively
tried different values for C. highest accuracy reached remains 0.73
decreased the data split to 60,40. which improved the accuracy to 0.76
implemented svm
tried linear kernel for decison boundary. accuracy 0.71
we have non-linear input data as seen so tried rbf, accuracy increasd to 0.91!
Feature corelation can be seen in "xplot.py". Here we can see the relevancy of each feature to the target.
also added reguralization C=3.0. accuraccy increased to 0.94! lower C lowers accuracy, higher C doesn't improve
going to implement Random forest and see if that improves anything

I have implemented a simple random forest model.
The model has already shown to be much better than svm and logistic regression
it has a 97% accuracy and the following confusion matrix
Confusion Matrix:
 [[183   5]
 [  8 182]]

 added a way to save the trained model so it can be loaded and be used for classification using Joblib
 also cut out 5 random rows of data from the csv file so that the model can be tested on unseen data
 Final result:
 Accuracy: 0.9840848806366048
Confusion Matrix:
 [[186   4]
 [  2 185]]
Classification Report:
               precision    recall  f1-score   support

           0       0.99      0.98      0.98       190
           1       0.98      0.99      0.98       187

    accuracy                           0.98       377
   macro avg       0.98      0.98      0.98       377
weighted avg       0.98      0.98      0.98       377



