#
# Author: Jamey Johnston
# Title: SciKit Learn Example with pickle
# Date: 2020/01/16
# Email: jameyj@tamu.edu
# Texas A&M University - MS in Analytics - Mays Business School
#

# Train models for Detecting Wine Quality
# Save model to file using pickle
# Load model and make predictions
#

# Import OS and set CWD
import os
from settings import APP_ROOT

import numpy as np
from numpy import loadtxt, vstack, column_stack
import xgboost
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd

# Import pickle to save ML models
import pickle

# Load the Wine Data
dataset = pd.read_csv("banknote_authentication.csv")

# Split the wine data into X (independent variable) and y (dependent variable)
X = dataset.iloc[:, 0:4].astype(float)
Y = dataset.iloc[:, 4].astype(int)

# Split wine data into train and validation sets
seed = 7
test_size = 0.3
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit model on Wine Training Data using eXtendeded Gradient Boosting
modelXGB = xgboost.XGBClassifier()
modelXGB.fit(X_train, y_train)

# Make predictions for Validation data
y_predXGB = modelXGB.predict(X_valid)
predictionsXGB = [round(value) for value in y_predXGB]

# Evaluate predictions
accuracyXGB = accuracy_score(y_valid, predictionsXGB)
print("Accuracy of eXtended Gradient Boosting: %.2f%%" % (accuracyXGB * 100.0))

# Create Dataset with Prediction and Inputs
predictionResultXGB = column_stack(([X_valid, vstack(y_valid), vstack(y_predXGB)]))

# Fit model on Wine Training Data using Random Forest save model to Pickle file
modelRF = RandomForestRegressor()
modelRF.fit(X_train, y_train)

# Make predictions for Validation data
y_predRF = modelRF.predict(X_valid)
predictionsRF = [round(value) for value in y_predRF]

# Evaluate predictions
accuracyRF = accuracy_score(y_valid, predictionsRF)
print("Accuracy of Random Forest: %.2f%%" % (accuracyRF * 100.0))

# Create Dataset with Prediction and Inputs
predictionResultRF = column_stack(([X_valid, vstack(y_valid), vstack(y_predRF)]))

# save model to file
pickle.dump(modelRF, open("../Assingment 2/src/static/BanknoteForged.pickleRF.dat", "wb"))

# Load model from Pickle file
loaded_modelRF = pickle.load(open("../Assingment 2/src/static/BanknoteForged.pickleRF.dat", "rb"))

# Predict a Wine Quality (Class) from inputs
loaded_modelRF.predict([[6.8, .47, .08, 2.2]])
