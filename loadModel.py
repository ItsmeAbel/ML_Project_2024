from joblib import load
import numpy as np

#unseen data
X_test= np.array([
    [53,1,0,110,335,0,1,143,1,3.0,1,1,3],  # First test sample
    [48,1,1,110,229,0,1,168,0,1.0,0,0,3],   # Second test sample
    [53,0,0,138,234,0,0,160,0,0.0,2,0,2],
    [58,1,1,125,220,0,1,144,0,0.4,1,4,3],
    [46,0,0,138,243,0,0,152,1,0.0,1,0,2]
])

#right answer: 00111
rf_model = load("randomForestModel.joblib")

#make predictions using the loaded model
pred = rf_model.predict(X_test)
print(pred)