#### Sparse matrix multiplication
### TF-IDF

import numpy as np
import scipy.sparse

tf = scipy.sparse.load_npz("tf_sparse.npz")
idf = scipy.sparse.load_npz("idf_sparse.npz")

tf_idf = tf.multiply(idf)

scipy.sparse.save_npz("tf_idf_final.npz", tf_idf)





