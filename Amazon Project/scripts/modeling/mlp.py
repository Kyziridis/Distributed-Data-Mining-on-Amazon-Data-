import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
import scipy
import sys
import os
import time
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping
import pickle

print("Starting timer")
start_time = time.time()

# Define Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=4)
model_checkpoint = keras.callbacks.ModelCheckpoint('mlp_checkpoint.hdf5',
					                               monitor='val_loss',
					                               verbose=0,
					                               save_best_only=True,
					                               save_weights_only=False,
					                               mode='auto',
					                               period=1)


# Transform ratings to one-hot vector
def encode_y(y):
    le = LabelEncoder()
    tags = le.fit_transform(y)
    y = np_utils.to_categorical(tags, 5)
    return y

# Batch generator to avoid meomry errors
def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = X.shape[0]/batch_size
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


#Load Data
first_time = time.time()
print("Loading Y ...")
y = np.load('ratings2.npy')
y = encode_y(y)
second_time = time.time()
print("Loaded Y")
print("it took", second_time - first_time)
print("Loading X ... ")
x = sp.load_npz('tf_idf_final.npz')
print("Loaded X")
third_time = time.time()
print("it took", third_time - second_time)

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=47)

# Saving splitted data
scipy.sparse.save_npz('x_test.spz', x_test)
np.save('y_test.npy',y_test)

# Split training data to training set and validation set for NN
xx_train, xx_test, xy_train, xy_test = train_test_split(x_train, y_train, test_size=0.10, random_state=47)
print('it took', time.time() - third_time)

print("Modeling now ...")
modeling_time = time.time()
model = Sequential()
model.add(Dense(10, input_shape=(5000,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['mae'])

history = model.fit_generator(generator=batch_generator(xx_train, xy_train, 32, True),
                    epochs=200,
                    validation_data=(xx_test, xy_test),
		    callbacks = [early_stopping, model_checkpoint],
			steps_per_epoch=20, shuffle=True)


print("Modeling took ...", time.time() - modeling_time)
print("ending ..." )
finish = time.time() - start_time
print('It all took ... ', finish)

# Save the Model
model.save(filepath=r'mlpAmazon.h5', overwrite=True)

# Save the History of training
with open('mlp_history.pickle', 'wb') as f:
    pickle.dump(history.history, f)

