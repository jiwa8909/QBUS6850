#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:04:56 2018

revised from
https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

QBUS6850 Lecture 6 Example 2
@author: Professor Junbin Gao, 
"""

import sklearn
import numpy  
from keras.preprocessing.text import one_hot
import keras

print("Numpy Version: " + numpy.__version__)
print("sklearn Version: " + sklearn.__version__)
print("keras Version: " + keras.__version__)

#This is a small doc with 18 words, but 17 words are different
doc = "This is a demo for one hot encoder. # The text (one document) can be converted to integer labels."

vocab_size = 30  # This is the maximal label for the code. 
                 # If it is too small, most likely different words can be mapped to the same lable
# filter is used to filter out those punctuations or symbols in text
labels = one_hot(doc, vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ')

print(labels)
        
vocab_size = 1000  # Now we set a larger size to reduce the collision

# filter is used to filter out those punctuations or symbols in text
labels = one_hot(doc, vocab_size,filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',lower=True, split=' ')

print(labels)        
 
