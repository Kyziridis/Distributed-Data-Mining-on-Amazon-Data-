import numpy as np
            
data = np.load('/local/amazon/codes.npy')

#we use a very smart function bincount:
terms = data[:, 1]
counts = np.bincount(terms, minlength=5559500 + 1) #f[n] is the frequency of term with code n
np.save('/local/amazon/chunk_counts.npy', counts)
