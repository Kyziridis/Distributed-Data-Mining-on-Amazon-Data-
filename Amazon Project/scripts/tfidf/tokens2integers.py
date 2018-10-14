# -*- coding: utf-8 -*-
"""
Created on Mon May 28 17:05:41 2018

@author: Wojtek
"""

#Convert tokenized reviews by replacing every token by an integer
#additionally, dump a dictionary: a list of tokens with their codes
import time
import numpy as np

input_file = "clean_reviews2.csv"
dict_file = "dictionary.txt"

token2ind = {}
max_code = -1
review_id = -1

start = time.time()

codes = []
with open(input_file) as f:
    for line in f:
        review_id = review_id + 1
        for w in line.split():
            if w in token2ind:
            	codes.append([review_id, token2ind[w]])
            else:
                max_code = max_code + 1
                token2ind[w] = max_code
                codes.append([review_id, token2ind[w]])
        
codes_array = np.asarray(codes, dtype=np.int32)

end = time.time()

print("\n=================\nDONE!")
print("Processed " + str(review_id + 1) + " reviews")
print("The number of unique terms = " + str(max_code + 1) + "\n")
print("Time taken to encode collection (in sec.): " + str(end - start))

start = time.time()
np.save('codes.npy', codes_array)
end = time.time()

print("Time taken to write .npy file of encoded collection (in sec.): " + str(end - start))

start = time.time()

#Dump the dictionary to a file:
f = open(dict_file, 'wt')
for token in token2ind:
    f.writelines(token + ", " + str(token2ind[token]) + "\n")
f.close()
    
end = time.time()

print('Time taken to dump dictionary of (token, code) (in sec.): ' + str(end - start))

#Should/could be implemented much more efficiently:  each review contains about 65 terms
#so 130 integers are sufficient to code it. Therefore pre-allocate memory (numpy array) of size:
#   (65+65)*80M= 130*80*4MB= 42GB and "write" codes directly to this array; 
#then save it in the numpy binary format.

#OR: instead of saving the first column; just remember the number of tokens per each document; 
#memory reduction: factor 2

#wojtek@boris:~/amazon$ wc 1M_words.txt
#  1000000  64240956 450963566 1M_words.txt
#
  

