#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import nltk 
import random
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import re
import json
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
import dask.bag as db
import dask.dataframe as dd
import time
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster, progress

p = ProgressBar()
p.register()


#cluster = LocalCluster(n_workers=2 , threads_per_worker=2 , ip= '145.107.189.23')
#c = Client()

# word stemmer
stemmer = LancasterStemmer()
random.seed(1)

def ImportData():
    """Function for importing the Json file dataset 
    as a pandas object and numpy array.
    Printing Datashape"""
        
    data = pd.read_json("sample_data.json", lines=True)
        
    # reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
    # asin - ID of the product, e.g. 0000013714
    # reviewerName - name of the reviewer
    # helpful - helpfulness rating of the review, e.g. 2/3
    # reviewText - text of the review
    # overall - rating of the product
    # summary - summary of the review
    # unixReviewTime - time of the review (unix time)
    # reviewTime - time of the review (raw)
    
    # Shape of data (79883, 9)
    print("\nData shape: ", np.shape(data))
    
    # Taking a subset for what we need
    data = data[['asin', 'reviewText', 'overall']]
        
    #Fix the data into numpy array class
    data1 = np.array(data)
    
    # Create a dask Dataframe
    df = dd.from_pandas(data,npartitions=6)
    
    # Create a bag from das kDataframe
    bag = df.reviewText.to_bag()
    
    #Output a pandas Data Frame(data) and a numpy array(data1)
    return data , data1 , bag , df

#Call the ImportData Function
p_data, np_data , bag , df= ImportData()

# Creating a sample of 70% from the initial dataset
x = random.sample(list(np.arange(np_data.shape[0])) , round(70/100*np_data.shape[0]))

# Splitting train/test sets 70/30 %
train_p_data , train_np_data = p_data.iloc[x,:] , np_data[x,:]
test_p_data , test_np_data = p_data.iloc[np.delete(np.arange(np_data.shape[0]),x),:] , np_data[np.delete(np.arange(np_data.shape[0]),x),:]

# Check if the split done correctly
train_np_data.shape[0] + test_np_data.shape[0] == np_data.shape[0]
    


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    
    #Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_review).lower()) 
    #letters_only = letters_only.lower()
    #Tokenize
    words = nltk.word_tokenize(letters_only)                           
   
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words and stem the correct words
    filtered = list(set(words) - stops)
    
    #meaningful_words = [stemmer.stem(w) for w in words if not w in stops]   
    meaningful_words = list(map(stemmer.stem , filtered))
    
    
    # Stem only the meaningful_words
# =============================================================================
#     stemmed_word = []
#     for i in meaningful_words:
#         stemmed_word.append(stemmer.stem(i))
# =============================================================================
    
    return( " ".join( meaningful_words ))   

# Parallel process instead of the for loop below.
#p = Pool(4)
#fasoula_parallel = p.map (review_to_words , train_np_data[:,1])



# Use dask to run it
start = time.time()
fasoula_dask = df.reviewText.map(review_to_words).compute()

#progress(fasoula_dask)
end = time.time()
print(end-start)


















print("\nCleaning/Stemming/Tokenizing the reviews....   >_\n")    
clean_train_reviews = []
for i in tqdm(range(train_np_data.shape[0])):
    # Call our function for each one, and add the result to the list of
    # clean reviews
    clean_train_reviews.append( review_to_words( train_np_data[:,1][i] ) )

###################################################################################

# Create bag of words
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000) 

# it fits the model and learns the vocabulary
# it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an 
# array
train_data_features = train_data_features.toarray()

# Create vocabulary
vocab = vectorizer.get_feature_names()

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# Initialize a Random Forest classifier with 100 trees
# Gini_impurity: Gini(E)=1−sum(p^2)
# Entropy: H(E)=-p*log(p)
forest = RandomForestClassifier(n_estimators = 100, criterion='entropy') 

print("\nTrain the RandomForest model, please wait..... >_")
forest = forest.fit( train_data_features, train_p_data['overall'])
print("\nDone\n")
print("\n")
############################################################


# Testing the model on the testset
# Create an empty list and append the clean reviews one by one
clean_test_reviews = [] 
print("\nCleaning the test_set and creating the test_bag_of_words    >_\n")
for i in tqdm(range(test_np_data.shape[0])):
    clean_test_reviews.append( review_to_words( test_np_data[:,1][i] ) )
    
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
test_data_features = test_data_features.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(test_data_features)

# Create Output Dataframe with id || Prediction || Truth
output = pd.DataFrame( data={"id":test_p_data["asin"], "Prediction":result, \
                             "Truth":test_p_data["overall"]} )

# Measure the percentage of correct predicted (True Positives)
correct = len(np.where(output.iloc[:,0] == output.iloc[:,1])[0])/output.shape[0]*100    
print("\nCorrect Predictions: %s%%"%np.round(correct,2) )

# Measure the error
error = output.iloc[:,0] - output.iloc[:,1]
RMSE = np.sqrt(np.mean(error**2))
print("\nRoot Mean Square Error: " + str(RMSE))
    
# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

matri = confusion_matrix(output.iloc[:,1] , output.iloc[:,0])






























