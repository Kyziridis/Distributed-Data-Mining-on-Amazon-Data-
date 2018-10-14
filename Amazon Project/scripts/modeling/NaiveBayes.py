import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import sys
import pickle
import scipy.sparse
import time

#latinum
x = scipy.sparse.load_npz('/local/amazon/modeling/tf_idf_final.npz')
#x = np.load('/local/amazon/modeling/tf_idf_final.npz')
y = np.load('/local/amazon/modeling/ratings2.npy')

#local
# x = np.random.rand(5000,100)
# y = np.random.randint(1,high=6,size=5000)

print('Data loaded')

s_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=47)
split_time = time.time() - s_time
print(' %s seconds' %split_time)
print('split over')

start_time = time.time()
nb = MultinomialNB()
nb.fit(x_train, y_train)
training_time = time.time() - start_time
print (' %s seconds '% training_time)
print('training over')

#local
# with open('/home/melidell024/Desktop/sddm/final/nb_model.p', 'wb') as file :
# 	pickle.dump(nb,file,protocol=pickle.HIGHEST_PROTOCOL)
#latinum
with open('/local/amazon/modeling/nb/nb_model.p', 'wb') as file :
	pickle.dump(nb,file,protocol=pickle.HIGHEST_PROTOCOL)

# sys.exit()
y_probs = nb.predict_proba(x_test)

# print(y_probs)

y_pred = y_probs[:,0] + 2*y_probs[:,1] + 3*y_probs[:,2] + 4*y_probs[:,3] + 5*y_probs[:,4]


#local
# np.save('/home/melidell024/Desktop/sddm/final/nb_pred.npy',y_pred)
#latinum
np.save('/local/amazon/modeling/nb/nb_pred.npy',y_pred)


#---------truncate-------------
y_pred[y_pred<1]=1
y_pred[y_pred>5]=5
y_pred = np.round(y_pred)

np.save('/local/amazon/modeling/nb/nb_pred_trunc.npy',y_pred)



# print(y_pred.shape)
mae = mean_absolute_error(y_test, y_pred)
print('MAE: ',mae)
rms = np.sqrt(mean_squared_error(y_test,y_pred))
print('RMS: ',rms)


with open("/local/amazon/modeling/nb/NB_Errors_trunc.txt", "w") as text_file:
    text_file.write("mean_absolute_error: %s" % mae)
    text_file.write("\n root_mean_squared_error: %s" % rms)
    text_file.write("\n training time: %s seconds" % training_time)

