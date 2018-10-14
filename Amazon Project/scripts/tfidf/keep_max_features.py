import numpy as np
            
nodes = [3, 4, 5, 6, 7, 8, 9, 12, 13, 14]
max_features = np.load('/scratch/Amazon_Project/max_features.npy')

# HERE IS THE MAIN TRICK: 
# We want to keep all the pairs with contain terms "to be kept"
# Let us find indicies of these pairs (as a boolean vectors with "TRUE" meaning "keep it")

chunk = np.load('/local/amazon/codes.npy')
pointers = round(chunk.shape[0] / 4) * np.arange(4, dtype=np.int64)
pointers = np.hstack((pointers, chunk.shape[0]))

for ind in range(len(pointers) - 1):
    chunk = chunk[pointers[ind]:pointers[ind + 1], :]
    chunk_max_features = chunk[np.in1d(chunk[:, 1], max_features)==True, :]    
    np.save('/local/amazon/codes' + str(ind + 1) + '_max_features.npy', chunk_max_features)
    del chunk
    del chunk_max_features
    if ind < 3:
        chunk = np.load('/local/amazon/codes.npy')

codes_max_features = np.load('/local/amazon/codes1_max_features.npy')
for ind in range(1, 4):
    chunk = np.load('/local/amazon/codes' + str(ind + 1) + '_max_features.npy')
    codes_max_features = np.vstack((codes_max_features, chunk))
    del chunk
np.save('/local/amazon/codes_max_features.npy', codes_max_features)
