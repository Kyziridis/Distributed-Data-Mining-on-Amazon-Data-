import numpy as np
import time

start = time.time()
            
nodes = ['03', '04', '05', '06', '07', '08', '09', '12', '13', '14']

count_sum = np.zeros(5559501)
for node in nodes:
    chunk_counts = np.load('/scratch/Amazon_Project/chunk_counts/chunk' + node + '_counts.npy')
    count_sum = count_sum + chunk_counts

sorted_codes = np.argsort(count_sum) # all terms sorted by their freqencies
n_terms_to_keep = 5000  
to_be_kept = sorted_codes[-n_terms_to_keep:] # codes of max features (sorted by freq.)

np.save('/scratch/Amazon_Project/max_features.npy', to_be_kept)

end = time.time()
print("Time taken to load count vectors of chunks (bincounts), perform reduce operation (sum of bincounts), sort count vector, and filter/select codes of max features (in sec.): " + str(end - start))
