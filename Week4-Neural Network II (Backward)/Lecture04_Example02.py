# -*- coding: utf-8 -*-

"""
Created on Wed Sep 27 10:46:19 2017

@author: wangch2
"""

# build NN manually

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    # apply the sigmoid activation fucntion
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    # derivative of the Sigmoid fucntion
    return np.exp(-z)/((1+np.exp(-z))**2)


nn_df = pd.read_csv("Lecture0802_Data.csv")
# nn_df = pd.read_csv("Lecture10_Data_1.csv")

X = nn_df.iloc[:, 0:-1]
y = nn_df.iloc[:,-1]

# standarized the data
X= X/np.amax(X, axis=0)
#y= np.reshape(y/np.amax(y, axis=0),(3,1))
y = y/np.amax(y, axis=0)
y = np.reshape(y.values, (3,1))

# initialize the weights      
input_layer_size= 2
hidden_layer_size= 3
output_layer_size= 1

# weight parameters
# define W(1): layer 1 to layer 2
np.random.seed(0)
W1= np.random.randn(input_layer_size, hidden_layer_size)
# define W(2): layer 2 to layer 3
W2= np.random.randn(hidden_layer_size, output_layer_size)    
# define learning rate
alpha= 3
# define the number of iterations
numIterations= 5000
# creat a list to save all loss function values
loss_list = list()

for i in range (0, numIterations):
    ######################################################      
    # forwad propagation fucntion       
    # forward propgation process based on lecture
    # calculate Z(2)
    z2= np.dot(X,W1)
    # calculate a(2)
    a2= sigmoid(z2)
    # calculate z(3)
    z3= np.dot(a2, W2)
    # calculate teh predition g_w_x (a(3))
    y_pred= sigmoid(z3)
    ######################################################   
    # calcualte the loss
    loss_value=  0.5*sum((y_pred-y)**2)   
    # append the loss into the list
    loss_list.append(loss_value)    
    #########################################################################
    # backward propagation fucntion      
    # calcualte loss fucntion derivative with respect to W1 and W2
    # calcualte the delta 3
    # this is a element wise multiplication
    delta3= np.multiply ((y_pred-y),sigmoid_prime(z3))   
    # calculate derivative dl_dW2
    # thsi is matrix normal multiplication
    dl_dW2= np.dot(a2.T, delta3)      
    # calcuatle delta 2
    # delta2= np.dot(delta3,W2.T)*sigmoid_prime(z2)
    delta2= np.multiply(np.dot(delta3,W2.T),sigmoid_prime(z2))
    # calculate derivative dl_dW1
    dl_dW1= np.dot(X.T, delta2)    
    #update the weights based on gradient descent
    W1= W1 - alpha*dl_dW1
    W2= W2 - alpha*dl_dW2


plt.plot(loss_list)












































































  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        