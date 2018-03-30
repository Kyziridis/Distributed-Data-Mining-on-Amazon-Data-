"""
TO DO: 1) Implementation of SVR with Gradient Descend.
       2) Random Forest with different number of features.
       3) Under-sampling, and fitting the models again (Not necessarily this week).

Prediction with baseline results in MAE of about 0.87
"""    
import pandas as pd
from nltk.corpus import stopwords
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model, svm
from sklearn.metrics import mean_absolute_error
from nltk.stem import PorterStemmer

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
                                 min_df=10, ngram_range=(1,3),\
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
    print('Training a Linear Regression Model.')
    svr = svm.SVR(C=1e3, gamma=0.1)
    model = svr.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return mean_absolute_error(y_test, y_pred)

def LinearReg(x_train, y_train, x_test, y_test):
    print('Training a Linear Regression Model.')
    reg = linear_model.LinearRegression()
    reg.fit (x_train, y_train)
    y_pred = reg.predict(x_test)
    return mean_absolute_error(y_test, y_pred)

stemmer = PorterStemmer()
# Load data
data = pd.read_json("sample_data.json", lines=True)
print('\nData Loaded.')

x_TF_train , x_TF_test , y_TF_train , y_TF_test = TF_IDF(data['reviewText'], data['overall'])
x_Count_train , x_Count_test , y_Count_train, y_Count_test = CountVect(data['reviewText'], data['overall'])

"""Linear Regression with Count Vectorizer"""
now = time.time()
print("\nLinear Reg. COUNT (MAE):", LinearReg(x_Count_train , y_Count_train, x_Count_test, y_Count_test))
print('Time taken to train: ' + str(time.time() - now))

"""Linear Regression with Tf-Idf Vectorizer"""
now = time.time()
print("\nLinear Reg. TF (MAE):", LinearReg(x_TF_train , y_TF_train, x_TF_test, y_TF_test))
print('Time taken to train: ' + str(time.time() - now))

"""Support Vector Regression with Count Vectorizer"""
now = time.time()
print("\nSVR TF (MAE):", SupportVectorReg(x_TF_train , y_TF_train ,x_TF_test, y_TF_test))
print('Time taken to train: ' + str(time.time() - now))

"""Support Vector Regression with Tf-Idf Vectorizer"""
now = time.time()
print("\nSVR COUNT (MAE):", SupportVectorReg(x_Count_train , y_Count_train, x_Count_test, y_Count_test))
print('Time taken to train: ' + str(time.time() - now))


#alaxa