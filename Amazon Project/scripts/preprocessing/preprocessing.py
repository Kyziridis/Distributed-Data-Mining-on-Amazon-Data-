import dask.dataframe as dd
import time
from dask.distributed import Client, LocalCluster
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
import os

cluster = LocalCluster(n_workers=20, threads_per_worker=2)
c = Client(cluster)

stemmer = PorterStemmer()
stops = stopwords.words("english")

def clean_reviews(raw_review):
    letters_only = re.sub("[^a-zA-Z]", " ", str(raw_review).lower())
    words = nltk.word_tokenize(letters_only)
    meaningful_words = [stemmer.stem(w) for w in words if not w in stops]
    return(" ".join(meaningful_words))

data = dd.read_csv("/local/amazon/item_dedup2.csv", blocksize=100*1024*1024) 
now = time.time()

print("Preprocessing...")
output = data.reviewText.map(clean_reviews)

out = output.compute(get=c.get)

print("Time taken for preprocessing (in sec.): ", time.time() - now)
out.to_csv("/local/amazon/clean_reviews2.csv", index=False)
