"""
Created on Sun 18 March 17:05:43 2018

@author: Professor Junbin Gao
 
Logistic Regression for Classification 
"""

import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
 
#df = pd.read_csv('Default.csv')
df = pd.read_csv('Lecture02_Default.csv')

# Convert the student category column to Boolean values
df.student = np.where(df.student == 'Yes', 1, 0)

# Get the features X and target/response y
X = df.iloc[:, 1:]   # X is a DataFrame
y = df.iloc[:, 0]    # Y is a Series

#Fit the Logistic Regression model

log_res = LogisticRegression()
log_res.fit(X, y)     # sklearn likes DataFrame and Series

"""
predict_proba() returns the probabilities of an observation 
belonging to each class. This is computed from the logistic 
regression function (see above).
"""
# Take out the first two cases to check the predicted class probability
prob = log_res.predict_proba(X.iloc[:2])
print("Probability of default for the first case: {0:.2f}%".format(prob[0,1] * 100))
print("Probability of default for the second case: {0:.2f}%".format(prob[1,1] * 100))

# Checking the assigned classes
outcome = log_res.predict(X.iloc[:2])
print("Assigned class for the first case: {0}".format(outcome[0]))
print("Assigned class for the second case: {0}".format(outcome[1]))

# Check all the data in training
from sklearn.metrics import confusion_matrix

pred_log = log_res.predict(X)
print(confusion_matrix(y,pred_log))
