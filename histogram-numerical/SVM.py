import dataload
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
import pickle 

x_train,x_test,y_train,y_test,label_index=dataload.dataloader()
# param_grid={'C':[0.1,1,10],'gamma':[0.0001,0.001,0.1],'kernel':['rbf','poly']}
# svc=svm.SVC(probability=True)
# model=GridSearchCV(svc,param_grid)
model=svm.SVC(kernel='poly',degree=3, C=10,gamma=1)
model.fit(x_train,y_train)
print('The Model is trained well ')

y_pred=model.predict(x_test) 
print("The predicted Data is :")
print(y_pred)
print("The actual data is:")
print(np.array(y_test))
print(f"The model is {accuracy_score(y_pred,y_test)*100}% accurate")

#model.save('color_svm.h5')  # creates a HDF5 file 'my_model.h5'
with open('label_index.pkl', 'wb') as f:
    pickle.dump(label_index, f)