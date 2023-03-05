import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


def dataloader():
	f=open("labelsmapping2.txt","w")
	flat_data_arr=[] #input array
	target_arr=[] #output array
	
	labels_to_aggregate = {
            'bej-2':'bej-1',
            'bej-3':'bej-1',
            'nokm-t':'sia',
            'ghah-t':'sia',
            'abi-t':'sorm'
        }
	Categories=glob("train/*")
	label_index={i.split("/")[-1]:Categories.index(i) for i in Categories}

	for i in Categories:
		f.write(str(Categories.index(i))+":"+i.split("/")[-1]+" ,")
		label=i.split("/")[-1]
		images=glob(i+"/*")
		for img in images:
			hist = np.load(img)
			flat_data_arr.append((hist-hist.mean())/hist.std())
			if label in labels_to_aggregate.keys():
				target_arr.append(label_index[labels_to_aggregate[label]])
			else:
				target_arr.append(label_index[label])
		print(f'loaded category:{i} successfully')

	flat_data=np.array(flat_data_arr)
	target=np.array(target_arr)
	df=pd.DataFrame(flat_data) #dataframe
	df['Target']=target
	df = df.sample(frac = 1)
	x_train=df.iloc[:,:-1] #input data 
	y_train=df.iloc[:,-1] #output data

	#test
	flat_data_arr=[] #input array
	target_arr=[] #output array
	
	for i in range(len(Categories)):
		Categories[i]=Categories[i].replace("train","test")

	for i in Categories:
		images=glob(i+"/*")
		for img in images:
			hist = np.load(img)
			label=i.split("/")[-1]
			flat_data_arr.append((hist-hist.mean())/hist.std())
			if label in labels_to_aggregate.keys():
				target_arr.append(label_index[labels_to_aggregate[label]])
			else:
				target_arr.append(label_index[label])
		print(f'loaded category:{i} successfully')

	flat_data=np.array(flat_data_arr)
	target=np.array(target_arr)
	df=pd.DataFrame(flat_data) #dataframe
	df['Target']=target
	df = df.sample(frac = 1)
	x_test=df.iloc[:,:-1] #input data 
	y_test=df.iloc[:,-1] #output data
	f.close()
	return x_train,x_test,y_train,y_test,label_index


