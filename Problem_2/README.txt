
------------------------------------Problem 2----------------------------------------------------------------------
To see step by step progress:
https://github.com/ItsmeAbel/ML_Project_2024/tree/main/Problem_2

I've gotten the data and started with data cleaning first. The csv data is filled with numerical, string, and float values, which means that some data cleaning is required.
The string values are catagorical and need to be encoded to numerical values.
We can use either Label encoder or One Hot Encoding
Label Encoding is good for ordianl data, while One Hot encoding is good for non-ordianl data.
With Label Encoding and multilinear regression model, i get the following result
mean_squared_error :  2335.762251930094
mean_absolute_error :  38.73413494045972
R2:  0.08190315105789459
which is more or less off by 1 root(MSE)
Result with one hot enconding:
Name: credit_score, Length: 45000, dtype: int64
mean_squared_error :  2227.70672219333
mean_absolute_error :  37.89635561614918
R2:  0.12437555649218801

which has slightly better result! So i'll be using OHE data. The R2 value however shows that label encoded model is worse
- tried svr as well and it took exceptionally long time to excute
-the svr uses standard scaler to scaled down the data during training
-it also uses the rbf kernel<------------------------------------gotta learn what it does
the result however isn't any better. it is the same as multilinear regression
Mean Squared Error (MSE): 2352.952252105177
Mean Absolute Error (MAE): 38.17397622064392
R-squared (R²): 0.07514643385304343

svr shows better result but takes too long to train. the improvment in accuracy isn't worth the tradeoff

i also implemented a decison tree and here are the result
Mean Squared Error: 4762.841777777778
R-squared Score: -0.8617762480146001
-the model is more or less off by 69, but the r2 value shows that decison tree performs worse than multilinear regression.

next up, random forest.
Initial Random forest implementation gives the following result:
Model Evaluation:
Mean Squared Error (MSE): 2461.65
Root Mean Squared Error (RMSE): 49.62
Mean Absolute Error (MAE): 39.75
R-squared (R2): 0.06
which isn't significantlly better than what we've tried so far

Next up: Neural Networks
Implemented a neural network model using the tensorflow and keras library
the model is pretty simple and uses the relu activation function
it has 4 layers including the input and output layers
i set the epoch size to 10 so i can better tweak and adjust it before choosing higher epoch sizes
so far, it gives more or less the same result as randomforest, but i will tweak the model so it becomes more accurate

NN tweaking: Tested out different variations for the design of the neural network. 
        Tried adjusting the amount of nodes in each layers, the amount of layers, and how the hidden layers are placed. 
        Currently it sort of looks like a romb
        -Tried implemnting a standard scaler and minmaxscaler
                - result without standard scaler:
                        Mean Absolute Error: 237.6289
                        Mean Squared Error: 87161.8585
                        Root Mean Squared Error (RMSE): 295.23
                        R2 Score: -32.3273
                - with standard scaler:
                        Mean Absolute Error: 39.4352
                        Mean Squared Error: 2406.4375
                        Root Mean Squared Error (RMSE): 49.06
                        R2 Score: 0.0799
        -Tried different activation functions. linear, relu and Leaky ReLU, Parametric Relu
            - linear is simply linear. High output values correspond high values with the activation function and vice versa.
            - similar to linear activation function but negetive values are given 0 value, meaning that those nerurons are treated as dead.
            -
            - tanh is best used for targets centered around 0, meaning that it spans between both posetive and negetive targets.
            -Sigmoid is best used for classification casses

        -Tried adjusting the epoch and batch size.
        Despite what i've tried so far, the highest accuracy score has yet to be improved
- I am going to check feature relevance incase we can decrease the feature size and elimnate unnecassary noises
 - I implemented code for analyzing feature importance using pandas and sklearn library
 - I can see some columns have little to no relevancy to model accuracy so i will try removing them
  - these columns seem to have very little relevancy:   person_home_ownership_OWN             0.008126
                                                        person_education_Bachelor             0.004867
                                                        person_education_Master               0.003140
                                                        person_education_Doctorate            0.002689
                                                        person_home_ownership_OTHER           0.001172
                                                        loan_percent_income                   0.000000
the entire relevance measurment
person_income                         0.231848
loan_amnt                             0.159127
loan_int_rate                         0.099213
person_emp_exp                        0.091153
person_age                            0.087466
cb_person_cred_hist_length            0.072343
person_education_High School          0.025959
previous_loan_defaults_on_file_Yes    0.019307
loan_intent_EDUCATION                 0.017543
loan_intent_MEDICAL                   0.016977
loan_intent_DEBTCONSOLIDATION         0.016914
loan_intent_PERSONAL                  0.016724
loan_status                           0.015989
loan_intent_VENTURE                   0.015679
previous_loan_defaults_on_file_No     0.015150
person_home_ownership_RENT            0.014113
person_home_ownership_MORTGAGE        0.013882
person_gender_female                  0.013625
person_gender_male                    0.013370
loan_intent_HOMEIMPROVEMENT           0.012521
person_education_Associate            0.011104
person_home_ownership_OWN             0.008126
person_education_Bachelor             0.004867
person_education_Master               0.003140
person_education_Doctorate            0.002689
person_home_ownership_OTHER           0.001172
loan_percent_income                   0.000000
dtype: float64

-- another feature relevancy measurment--
person_emp_exp                        0.025407
person_age                            0.021794
cb_person_cred_hist_length            0.020340
previous_loan_defaults_on_file_No     0.018377
person_education_High School          0.015358
previous_loan_defaults_on_file_Yes    0.013788
person_education_Doctorate            0.012869
person_gender_female                  0.006092
person_education_Master               0.005321
person_education_Bachelor             0.004606
loan_status                           0.004130
person_education_Associate            0.004031
loan_intent_DEBTCONSOLIDATION         0.002710
person_gender_male                    0.002468
loan_amnt                             0.002372
loan_int_rate                         0.002129
person_home_ownership_OWN             0.001838
loan_intent_VENTURE                   0.001648
person_home_ownership_RENT            0.000148
loan_intent_MEDICAL                   0.000000
loan_intent_PERSONAL                  0.000000
loan_percent_income                   0.000000
loan_intent_HOMEIMPROVEMENT           0.000000
loan_intent_EDUCATION                 0.000000
person_home_ownership_MORTGAGE        0.000000
person_income                         0.000000
person_home_ownership_OTHER           0.000000


I tried picking the 10 most relevant features and used them in my NN, but result remains the same
Im going to plot the result in comparison to the prediction and see if i can spot out any errors
 - tried it, doesn't help much
 Result with 10 most relevant features:
 Mean Absolute Error: 38.9056
Mean Squared Error: 2471.2929
Root Mean Squared Error (RMSE): 49.71
R2 Score: 0.0551

Tried implementing a normalizer and/or dropout reguralization techiques incase there is an overfitting issue.
- Result remains the same!

- i also used a corelation matrix to visualize the feature corelation. This is found in "xplot.py"
Here we can see, on the credit_score row, that cells with 0 value have little to no relevancy to the models accuracy
and therefore can be removed.
i hand picked the 7 modt relevant features and tried with them
reuult: 
Mean Absolute Error: 42.5272
Mean Squared Error: 2662.8781
Root Mean Squared Error (RMSE): 51.60
R2 Score: -0.0182

Created EDA.py and did more analysis on the data set, which admittedly i should have done more thoroughly from the start
Using the EDA i can get better insight into the data, which i can use to better prepare the training data.
The new csv file is named "OutlierfreeOHE.csv"
I have now done a thorough data analysis
I sperated the data into numerical and catagorical data.
I removed and/or grouped outliers from both types of data. After removing the ouliers, the rows of data decreased from ~45k to ~28k
I then used one hot encoding to encode the data
Thereafter, i used a correlation matrix to identify which the most relevant columns. This is found in the "xplot.py" file
I then trained the NN with the new csv file and got the following results
        Mean Absolute Error: 37.6565
        Mean Squared Error: 2088.6528
        Root Mean Squared Error (RMSE): 45.70
        R2 Score: 0.0765
which is slightly better.
I tried using the parametric ReLU again with the new file. File "PReLUNN.py"
Result: Mean Absolute Error: 38.2412
        Mean Squared Error: 2433.6684
        Root Mean Squared Error (RMSE): 49.33
        R2 Score: -0.0761
which isn't any better.
I also tried the new csv file with random forest but got no better result.
I also tried removing the catagorical data outliers instead of grouping them but didn't show any improvments.
I have concluded that this is as high accuraccy as i can get for now. The only thing left to do is to pick a higher epoch and retrain the model.

FInal result with 100 epochs but early stopping: Mean Absolute Error: 36.9577
                                                                Mean Squared Error: 2071.9989
                                                                Root Mean Squared Error (RMSE): 45.52
                                                                R2 Score: 0.0838
The final model is saved in "NNModel.joblib"