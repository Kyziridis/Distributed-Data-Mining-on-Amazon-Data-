import os
import numpy as np
from keras.models import load_model
import sys
import scipy.sparse as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import time


all = time.time()
# Load the model
print("Loading the model ...")
one = time.time()
my_model = load_model('mlpAmazon.h5')
print("It took ", time.time() - one)

print("Loading the test set ...")
two = time.time()
x_test = sp.load_npz('x_test.npz')
print("It took ", time.time() - two)

print("Predicting the test set ...")
three = time.time()
y_pred = my_model.predict(x_test)
print("It took ", time.time() - three)

print("Saving the predictions ...")
four = time.time()
np.save('y_pred.npy', y_pred)
print("It took ", time.time() - four)

print("Loading test set labels ...")
five = time.time()
y_true = np.load("y_test_int.npy")
print("It took ", time.time() - five)

categories = [1,2,3,4,5]

print("Calculating weighted average for regression ...")
six = time.time()
mul = categories * y_pred
y_pred_class = np.sum(mul, axis = 1)
print("It took ", time.time() - six)

print("Mean absolute error ...")
seven  = time.time()
mae = mean_absolute_error(y_true, y_pred_class)
print("MAE: ", mae)
print("It took ", time.time() - seven)

print("Root mean sqaure error ...")
eight  = time.time()
rmse = sqrt(mean_squared_error(y_true, y_pred_class))
print("RMSE: ", rmse)
print("It took ", time.time() - eight)

print("All of it took ", time.time() - all)

