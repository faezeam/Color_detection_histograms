import pickle 
import dataload
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
import pickle 


with open('label_index.pkl', 'rb') as f:
    label_index = pickle.load(f)

model = tf.keras.models.Sequential()
model=model.load('color_fullyconnected.h5')

