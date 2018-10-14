# Truncate Metrics


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.sparse
import time
import pickle

# Load tf-idf sparse matrix
start = time.time()
x = scipy.sparse.load_npz("tf_idf_final.npz")
print("Loading tf-idf sparse took " + str(time.time() -start))


# Load Ratings
start = time.time()
y = np.load("ratings2.npy")
print("Loading Ratings took " + str(time.time() - start))


# Split train/test
print("Splitting train/test set")
now = time.time()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=47)
print("Splitting took " + str(time.time() - now))

# Baseline
estimation = np.mean(y_train)
estimation = np.round(estimation)
pred = np.full(len(y_test), estimation)

print("BaseLine Metrics/n")
print("MAE  : " + str(mean_absolute_error(y_test, pred)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_test, pred))))
print("-------------------------")

# LM truncated
x = np.load("LM_pred.npy")
print("Pred loaded")
trunc = x
trunc[trunc<1] = 1
trunc[trunc>5] = 5
trunc = np.round(trunc)

print("LM-Truncated Metrics")
print("MAE  : " + str(mean_absolute_error(y_test, trunc)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_test, trunc))))
print("-------------------------")
print("ANTE GEIA POUTANES")
np.save("truncated_LM_pred", trunc)
