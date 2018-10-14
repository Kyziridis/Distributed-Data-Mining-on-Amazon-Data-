import numpy as np
import time

start = time.time()

data = np.load('codes.npy')

end = time.time()

print("Time taken to load encoded data: " + str(end - start))

start = time.time()

n_splits = 10
data_splitted = np.array_split(data, n_splits)

split_indices = np.zeros(n_splits - 1, dtype=np.int64)
for split in range(len(data_splitted) - 1):
    split_indices[split] = np.where(data[:, 0] == data_splitted[split][:, 0][-1])[0][-1]
print("Done - Data splitted in " + str(n_splits) + " chunks.")

data_splitted = np.array_split(data, split_indices + 1)

nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

if n_splits == len(nodes):
    for idx, split in enumerate(data_splitted):
        np.save('/local/amazon/codes_data/codes' + str(nodes[idx]), split)
    print("Done - " + str(n_splits) + " chucks saved.")
end = time.time()

print("Tike taken to split and dump encoded data: " + str(end - start))
