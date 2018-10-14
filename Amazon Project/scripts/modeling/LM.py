# Linear Model for tf-idf


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import scipy.sparse
import time
import pickle

# Set the time for the whole script
general_time = time.time()

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

# Train the model#################
# Create linear regression object
regr = LinearRegression(n_jobs=8)

# Train the model using the training sets
print("Start training.... please wait")
now = time.time()
regr.fit(X_train, y_train)
print("Training 8-threads took " + str(time.time()-now))

save = time.time()
filename = 'LM_final.sav'
pickle.dump(regr, open(filename, 'wb'))
print("Save model to disk took" + str(time.time() - save))
######

# Make predictions using the testing set
now = time.time()
y_pred = regr.predict(X_test)
np.save("LM_pred",y_pred)
print("Saving y_pred took " + str(time.time() - now))
##########################

print("-----Metrics------")
print("MAE  : " + str(mean_absolute_error(y_test, y_pred)))
print("RMSE : " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
print("Variance score : " + str(r2_score(y_test, y_pred)))
print("Time for the whole process : " + str(time.time() - general_time))
print("ANTE GEIA POUTANES")










 
