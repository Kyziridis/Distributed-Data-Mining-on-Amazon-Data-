import pickle
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import time
import scipy.sparse

print("Calculating number of reviews...")
tf_indices = np.load('/local/amazon/tf_indices.npy')
num_reviews = len(np.unique(tf_indices[:, 0]))
print("Done.")

start = time.time()
nodes = [12, 13, 3, 4, 5, 6, 7, 8, 9, 14]
for node in nodes:
    with open('/local/amazon/idf/idf' + str(node) + '.pickle', 'rb') as handle:
        idf = pickle.load(handle)
    if node == 12:
        idf_sum = idf
    else:
        for key, value in idf.items():
            idf_sum[key] += idf[key]     

for key in idf_sum:
    idf_sum[key] = np.log2(num_reviews / idf_sum[key])

end = time.time()
print("Time taken to sum document frequency values across chunks (reduce operation) and calculate idf (in sec.): " + str(end - start))

start = time.time()
with open('/local/amazon/idf.pickle', 'wb') as handle2:
    pickle.dump(idf_sum, handle2, protocol=pickle.HIGHEST_PROTOCOL)
end = time.time()
print("Time taken to dump dictionary of idf values (in sec.): " + str(end - start))

start = time.time()

keys = []
for key in idf_sum:
    keys.append(int(key))
max_key = np.max(np.array(keys))

idf_array = np.zeros(max_key + 1, dtype=np.int32)
for key, value in idf_sum.items():
    idf_array[int(key)] = value
idf_array = idf_array[idf_array > 0]

idf_sparse = scipy.sparse.csc_matrix(idf_array)
scipy.sparse.save_npz('/local/amazon/idf_sparse.npz', idf_sparse)

end = time.time()
print("Time taken to construct sparse idf matrix (in sec.): " + str(end - start))

#start = time.time()
#dict_vect = DictVectorizer(sparse=True)
#idf_sparse = dict_vect.fit_transform(idf_sum)
#sparse.save_npz("/local/amazon/idf_sparse.npz", idf_sparse)
#end = time.time()
#print("Time taken to convert dictionary of idf values to sparse matrix and dump it (in sec.) " + str(end - start)) 



