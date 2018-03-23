import numpy as np
import pandas as pd
import nltk
import random
from nltk.corpus import stopwords
import re
import dask.bag as db
import dask.dataframe as dd
import time
from dask.distributed import Client, LocalCluster, progress
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from nltk.stem import *

stemmer = LancasterStemmer()

# Load data
data = pd.read_json("sample_data.json", lines=True)



# Preprocess with stemming and removing stop-words
def review_to_words( raw_review ):
    #Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_review).lower()) 
    #letters_only = letters_only.lower()
    #Tokenize
    words = nltk.word_tokenize(letters_only)                           
   
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words and stem the correct words
    meaningful_words = [stemmer.stem(w) for w in words if not w in stops]   
       
    return( " ".join( meaningful_words ))



# Call the preprocess
now = time.time()
pdata = list(map(review_to_words , data['reviewText']))
print(' Time for preprocess: ' + str(time.time() - now))
print('>_ Support GNU/Linux >_')

def TF_IDF(x , y):
	"""Function for making TF-IDF
	   training a model with vectorizer
	   split the dataset
	   Inputs: x=column with reviews , y=column with overall
	    """
	
	#TF-IDF
	vectorizer = TfidfVectorizer(stop_words = None,\
		analyzer = "word",min_df=10,\
		ngram_range=(1,2),max_features=5000 )

	# Train the model
	train_data_features = vectorizer.fit_transform(x)

	# Learn vocabulary
	vocab = vectorizer.get_feature_names()


	# Split dataset 70/30
	X_train, X_test, y_train, y_test = train_test_split(train_data_features, y,\
		 test_size=0.3, random_state=0)

	return X_train, X_test, y_train, y_test

x_TF_train , x_TF_test , y_TF_train , y_TF_test = TF_IDF(pdata , data['overall'])

def CountVect(x,y):
	"""Function for making CountVectorizer
	   Inputs: x= column with reviews , y=column with overall
	 """

	vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

	# it fits the model and learns the vocabulary
	# it transforms our training data
	# into feature vectors. The input to fit_transform should be a list of strings.
	train_data_features = vectorizer.fit_transform(x)

	vocab = vectorizer.get_feature_names()

	X_train, X_test, y_train, y_test = train_test_split(train_data_features, y,\
	 	test_size=0.3, random_state=0)

	return X_train, X_test, y_train, y_test

x_Count_train , x_Count_test , y_Count_train, y_Count_test = CountVect(pdata , data['overall'])



def Regression(x_train, y_train , x_test , y_test ):

	# Linear Regression
	print('Regresio')
	reg = linear_model.LinearRegression()
	reg.fit (x_train, y_train)
	result = reg.predict(x_test)
	print("MAE: " + str(mean_absolute_error(y_test, result)) )