import numpy as np

codes = np.load('/local/amazon/codes_max_features.npy')
	
indices = np.cumsum(np.bincount(codes[:, 0]))
indices = indices[indices > 0]
indices = np.hstack((0, indices))

sum_unique_terms = 0
for index in range(len(indices) - 1):
    chunk = codes[indices[index]:indices[index + 1], :]
    sum_unique_terms += len(np.unique(chunk[:, 1]))
    
left_index = 0
freq_vector = np.zeros(sum_unique_terms, dtype=np.int16)
for index in range(len(indices) - 1):
    chunk = codes[indices[index]:indices[index + 1], :]
    if chunk.shape[0] == 0:
        continue
    tf = {}
    for term in chunk[:, 1]:
        if term in tf:
            tf[term] += 1 
        else:
            tf[term] = 1
    unique_terms = np.unique(chunk[:, 1])
    right_index = left_index + len(unique_terms)
    codes[left_index:right_index, 0] = chunk[0, 0]
    codes[left_index:right_index, 1] = unique_terms
    freq_vector[left_index:right_index] = np.fromiter(tf.values(), dtype=np.int16)
    left_index = right_index

codes = codes[:freq_vector.shape[0], :]

np.save('/local/amazon/tf_indices2.npy', codes)
np.save('/local/amazon/tf_values2.npy', freq_vector)
