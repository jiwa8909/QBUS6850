# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 12:10:09 2021

Also see an example at
https://medium.com/analytics-vidhya/linear-regression-with-pytorch-147fed55f138

QBUS6850 Lecture 2 Example 2
@author: Professor Junbin Gao
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

car_data_df = pd.read_csv('Lecture2_Regression.csv')

dim= car_data_df.shape
n_row= dim[0]
n_col= dim[1]
#car_price= car_data.iloc[:,1]

car_price= car_data_df['Price']
odometer= car_data_df['Odometer']

y_data = np.reshape(np.array(car_price), (len(car_price), 1))    
x_data = np.reshape(np.array(odometer), (len(odometer), 1))


# Your model must be inherited from the pytorch module, with which pytorch 
# can do automatic BP for gradients.  This is specially for 1 dimensionality input and output
class myLinearRegression(torch.nn.Module):
    def __init__(self):
        super(myLinearRegression, self).__init__()
        self.beta0 = torch.nn.Parameter(torch.tensor(20.0))
        self.beta1 = torch.nn.Parameter(torch.tensor(-0.1))
        
    def forward(self, x):
        out = self.beta0 + self.beta1 * x
        return out
    

learningRate = 0.001 
epochs = 1000

model = myLinearRegression()
# Use the standard stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

# Convert data to torch tensors
inputs = torch.from_numpy(x_data).type(torch.float32)
targets =torch.from_numpy(y_data).type(torch.float32)

for epoch in range(epochs):
    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    #loss = criterion(outputs, targets)
    loss = 0.5*torch.mean((outputs - targets).pow(2.0))
    print(loss)
    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
    
# Print intercept and coefficients
print("\nThe estimated model parameters are")
print(model.beta0)    # This is the intercept \beta_0 in our notation
print(model.beta1)         # This is \beta_1 in our notation


# plot the fitted linear regression line
with torch.no_grad(): # we don't need gradients in the testing phase
    x_temp = np.reshape(np.linspace(np.min(x_data), np.max(x_data), 50), (50,1))
    y_temp = model(torch.from_numpy(x_temp).type(torch.float32)).numpy()
     
plt.figure()

plt.plot(x_temp,y_temp)
plt.scatter(odometer,car_price,label = "Observed Points", color = "red")




# Way 2:
# Use some standard building blocks
class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.001 
epochs = 15000

model = linearRegression(inputDim, outputDim)
# Use the built-in Least Square Loss
criterion = torch.nn.MSELoss() 
# Use the standard stochastic gradient descent optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, momentum=0.82)

for epoch in range(epochs):    
    optimizer.zero_grad()
    outputs = model(inputs)
    # get loss for the predicted output
    loss = criterion(outputs, targets)
    print(loss) 
    loss.backward() 
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))
    
# Print intercept and coefficients
print("\nThe estimated model parameters are")
print(model.linear.bias)    # This is the intercept \beta_0 in our notation
print(model.linear.weight)         # This is \beta_1 in our notation

# plot the fitted linear regression line
with torch.no_grad(): # we don't need gradients in the testing phase
    x_temp = np.reshape(np.linspace(np.min(x_data), np.max(x_data), 50), (50,1))
    y_temp = model(torch.from_numpy(x_temp).type(torch.float32)).numpy()


plt.figure()

plt.plot(x_temp,y_temp)
plt.scatter(odometer,car_price,label = "Observed Points", color = "red")
 

"""
optimizer.zero_grad()
output = model(input)
loss = loss_fn(output, target)
loss.backward()
optimizer.step()
"""