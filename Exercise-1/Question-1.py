"""
In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras


model = # Your Code Here#
model.compile(# Your Code Here#)
xs = # Your Code Here#
ys = # Your Code Here#
model.fit(# Your Code here#)
print(model.predict([7.0]))
