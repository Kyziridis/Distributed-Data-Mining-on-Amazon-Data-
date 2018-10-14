import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
import time

nodes = [12, 13, 3, 4, 5, 6, 7, 8, 9, 14]
ratings = np.load('/local/amazon/ratings.npy')

start = time.time()
    
for node in nodes:
    if node == 12:
        tf_indices_matrix = np.load('/local/amazon/tf/tf_indices' + str(node) + '.npy')
        tf_values_matrix = np.load('/local/amazon/tf/tf_values' + str(node) + '.npy')
        tf_values_matrix = tf_values_matrix.astype(np.float32)
    else:
        tf_indices = np.load('/local/amazon/tf/tf_indices' + str(node) + '.npy')
        tf_values = np.load('/local/amazon/tf/tf_values' + str(node) + '.npy')
        tf_values = tf_values.astype(np.float32)
        tf_indices_matrix = np.vstack((tf_indices_matrix, tf_indices))
        tf_values_matrix = np.hstack((tf_values_matrix, tf_values))

end = time.time()

print("Time taken to load and concatenate tf indices and values (in sec.): " + str(end - start))

tf_values_matrix = np.log2(tf_values_matrix) + 1
np.save('/local/amazon/tf_indices.npy', tf_indices_matrix)
np.save('/local/amazon/tf_values.npy', tf_values_matrix)

start = time.time()

tf_sparse = csr_matrix((tf_values_matrix, (tf_indices_matrix[:, 0], tf_indices_matrix[:, 1])), shape=(np.max(tf_indices[:, 0]) + 1, np.max(tf_indices[:, 1]) + 1))

end = time.time()

print("Time taken to construct sparse matrix (in sec.): " + str(end - start))

start = time.time()

nonzero_rows = tf_sparse.getnnz(1) > 0
tf_sparse = tf_sparse[nonzero_rows, :]
tf_sparse = tf_sparse.tocsc()
nonzero_columns = tf_sparse.getnnz(0) > 0
tf_sparse = tf_sparse[:, nonzero_columns]

end = time.time()

print("Time taken to select non-zero rows and columns (in sec.): " + str(end - start))

start = time.time()
scipy.sparse.save_npz('/local/amazon/tf_sparse.npz', tf_sparse)
end = time.time()
print("Time taken to dump sparse matrix (in sec.): " + str(end - start))

ratings = ratings[nonzero_rows]
np.save('/local/amazon/ratings2.npy', ratings)





