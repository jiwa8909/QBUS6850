#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu April 22 15:40:04 2021

@author: Professor Junbin Gao
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# fix random seed for reproducibility
np.random.seed(7)
torch.manual_seed(1)

# load pima indians dataset
dataset = np.loadtxt("Lec5_pima-indians-diabetes.csv", delimiter=",")

# split into input (X) and output (Y) variables
X = torch.Tensor(dataset[:,0:8])
Y = torch.Tensor(dataset[:,8]).unsqueeze(1)


input_size = 8
hidden_sizes = [12, 8]    
output_size = 1

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.Sigmoid())
    
# Prepare for training
epochs = 150
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)  # Setting regularizer 
criterion = nn.BCELoss()
losses = []

for i in range(epochs):
    print('Epoch:  ',i+1)
    model.zero_grad()
    model.train()
    y = model(X)
    loss = criterion(y, Y)
    loss.backward()
    optimizer.step()        
    losses.append(loss.data.numpy())

plt.plot(losses)
