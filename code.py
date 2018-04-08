"""
TO DO: 1) Implementation of SVR with Gradient Descend.
       2) Random Forest with different number of features.
       3) Under-sampling, and fitting the models again (Not necessarily this week).

Prediction with baseline results in MAE of about 0.87
"""
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import sys

def tokenization(x):
    return x.split(" ")

def TF_IDF(x, y, num_features = 5000):
	"""
    Function for creating a matrix of Tf-Idf values, and then splitting in training and test sets.
    Inputs: x = list of reviews, as returned from review_to_words()
            y = column with overall rating
            num_features = Number of columns for the Tf-Idf matrix
    Output: Training and test sets containing Tf-Idf values
	"""

	vectorizer = TfidfVectorizer(stop_words='english',\
                                 token_pattern = "\w*[a-z]\w*",\
                                 tokenizer=tokenization,\
                                 analyzer="word",\
                                 min_df=10, ngram_range=(1,2),\
                                 max_features=num_features)
	train_data_features = vectorizer.fit_transform(x)
	vocab = vectorizer.get_feature_names()
	x_train, x_test, y_train, y_test = train_test_split(train_data_features, y,\
                                                       test_size=0.3, random_state=0)
	return x_train, x_test, y_train, y_test

def CountVect(x, y, num_features = 5000):
	"""
    Function for creating a matrix of Frequencies (counts), and then splitting in training and test sets.
    Inputs: x = list of reviews, as returned from review_to_words()
            y = column with overall rating
            num_features = Number of columns for the Tf-Idf matrix
    Output: Training and test sets containing frequencies
	"""
	vectorizer = CountVectorizer(analyzer = "word",   \
                                 token_pattern = "\w*[a-z]\w*",\
                                 tokenizer=tokenization,\
                                 preprocessor = None, \
                                 min_df = 10,\
                                 stop_words = 'english',   \
                                 max_features = num_features)

	train_data_features = vectorizer.fit_transform(x)
	vocab = vectorizer.get_feature_names()
	x_train, x_test, y_train, y_test = train_test_split(train_data_features, y,\
                                                        test_size=0.3, random_state=0)

	return x_train, x_test, y_train, y_test

def SupportVectorReg(x_train, y_train, x_test, y_test):
    print('Training an SVR Model.')
    svr = svm.SVR(C=1e3, gamma=0.1)
    model = svr.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_absolute_error(y_test, y_pred)

def SDG_SupportVectorReg(x_train, y_train, x_test, y_test):
    print('Training an SDG_SVR Model.')
    sdg_svr = linear_model.SGDClassifier()
    model = sdg_svr.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_absolute_error(y_test, y_pred)

def LinearReg(x_train, y_train, x_test, y_test):
    print('Training a Linear Regression Model.')
    reg = linear_model.LinearRegression()
    reg.fit (x_train, y_train)
    y_pred = reg.predict(x_test)
    return mean_absolute_error(y_test, y_pred)

def RandomForestReg(x_train, y_train, x_test, y_test):
	print('Training a Random Forest Regression Model.')
	reg = RandomForestRegressor(max_features=100,n_estimators = 50,max_depth=100,n_jobs = 4)
	reg.fit (x_train, y_train)
	y_pred = reg.predict(x_test)
	return mean_absolute_error(y_test, y_pred)



#Baseline function computes the MAE with the mean of data['overall'] as a prediction
def Baseline(x,y):
	base_test = np.mean(y)
	result = np.full(len(y),base_test)
	print("Baseline Mae:" + str(mean_absolute_error(y,result)))






#Select the feature extraction method and the regression method
#features='count' or 'tf-idf'
#regressor='linear','svr' or 'rf'
def Regression(features,regressor):
	# Tf-Idf Vectorizer
	if features=="tf-idf":
		print('Feature Extraction: Tf-Idf')
		x_train , x_test , y_train , y_test = TF_IDF(data['reviewText'], data['overall'])
	# Count Vectorizer
	elif features=="count":
		print('Feature Extraction: CountVect')
		x_train , x_test , y_train, y_test = CountVect(data['reviewText'], data['overall'])

	if regressor=="linear":
		# """Linear Regression
		now = time.time()
		print("\nLinear Reg. (MAE):", LinearReg(x_train , y_train, x_test, y_test))
		print('Time taken to train: ' + str(time.time() - now))
	elif regressor=="svr":
		# """Support Vector Regression
		now = time.time()
		print("\nSVR (MAE):", SupportVectorReg(x_train , y_train ,x_test, y_test))
		print('Time taken to train: ' + str(time.time() - now))
	elif regressor=="rf":
		# """Random Forest Regression"""
		now = time.time()
		print("\nRFR (MAE):", RandomForestReg(x_train , y_train ,x_test, y_test))
		print('Time taken to train: ' + str(time.time() - now))
	elif regressor=="sdg_svr":
		# """SDG Support Vector Regression
		now = time.time()
		print("\nSVR (MAE):", SDG_SupportVectorReg(x_train , y_train ,x_test, y_test))
		print('Time taken to train: ' + str(time.time() - now))

if __name__=='__main__':

	# Load data
	data = pd.read_json("sample_data.json", lines=True)
	print('\nData Loaded.')

	#Calling the baseline
	Baseline(data['reviewText'],data['overall'])

	#Doing the Regression
	Regression("tf-idf","sdg_svr")
	Regression("count","sdg_svr")
