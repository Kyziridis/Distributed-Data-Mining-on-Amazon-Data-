import numpy as np
import scipy.sparse
import sklearn.model_selection
import sklearn.linear_model
import pickle
import time

y = np.load('/local/amazon/modeling/ratings2.npy')
x = scipy.sparse.load_npz('/local/amazon/modeling/tf_idf_final.npz')

print("Data loaded.")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, random_state=47)

print("Data spliteed into training and test.")

start = time.time()
logreg = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=300, verbose=1, n_jobs=8)
logreg.fit(x_train, y_train)
end = time.time()

print("Time taken to train logistic regression model (in sec.): " + str(end-start))

proba_predictions = logreg.predict_proba(x_test)

rating_values = np.array([1, 2, 3, 4, 5])

weighted_proba = rating_values * proba_predictions
predictions = np.sum(weighted_proba, axis=1)

np.save('/local/amazon/modeling/predictions_logreg.npy', predictions)
np.save('/local/amazon/modeling/y_test_logreg.npy', y_test)

filename = 'logreg_model.sav'
pickle.dump(logreg, open(filename, 'wb'))
