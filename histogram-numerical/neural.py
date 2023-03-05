import dataload
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle 


x_train,x_test,y_train,y_test,label_index=dataload.dataloader()
#x_train_norm=(x_train-x_train.mean())/x_train.std()
#x_test_norm=(x_test-x_test.mean())/x_test.std()
y_train_enc = tf.keras.utils.to_categorical(y_train)
y_test_enc = tf.keras.utils.to_categorical(y_test)

in_dim = 768

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(512, input_dim=in_dim, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, input_dim=in_dim, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(23, activation=tf.nn.softmax))


sgd=tf.keras.optimizers.SGD()
model.compile(optimizer=sgd ,loss="categorical_crossentropy", metrics=["accuracy"])
EP = 25
history = model.fit(x=x_train, y=y_train_enc, epochs=EP, validation_data=(x_test, y_test_enc))
model.save('color_fullyconnected.h5')  # creates a HDF5 file 'my_model.h5'
with open('label_index.pkl', 'wb') as f:
    pickle.dump(label_index, f)