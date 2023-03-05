import dataload
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
import pickle 
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


x_train,x_test,y_train,y_test,label_index=dataload.dataloader()

#one-hot encode target column
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



model = keras.models.Sequential([
    keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape = [64,64,3]),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(32, (5, 5), activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, (5, 5), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(23, activation ='softmax')
])

model.compile(optimizer='adam',
             loss = 'categorical_crossentropy',
             metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss',
                                            patience=3,
                                            restore_best_weights=True)



model.fit(x_train,y_train, epochs=30, validation_data=(x_test,y_test), callbacks=callback)