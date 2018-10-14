import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

predictions = np.load('/local/amazon/modeling/predictions_logreg.npy')
y_test = np.load('/local/amazon/modeling/y_test_logreg.npy')

predictions = np.round(predictions)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print("Logistic Regression; RMSE: " + str(np.sqrt(mse)))
print("Logistic Regression; MAE: " + str(mae))

#mse = np.sqrt(np.sum((y_test - predictions) ** 2) / len(y_test))
#mae = np.sum(abs(y_test - predictions)) / len(y_test)

