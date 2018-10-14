import numpy as np
import pickle

codes = np.load('/local/amazon/codes_max_features.npy')

indices = np.cumsum(np.bincount(codes[:, 0]))
indices = indices[indices > 0]
indices = np.hstack((0, indices))

idf = {}

for index in range(len(indices) - 1):
    chunk = codes[indices[index]:indices[index + 1], :]
    terms = np.unique(chunk[:, 1])
    for term in terms:
        if term in idf:
            idf[term] += 1 
        else:
            idf[term] = 1
            
with open('/local/amazon/idf.pickle', 'wb') as handle:
    pickle.dump(idf, handle, protocol=pickle.HIGHEST_PROTOCOL)
