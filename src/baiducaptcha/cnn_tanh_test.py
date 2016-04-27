from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  
import os
import Image
import time

from keras.callbacks import ModelCheckpoint 
from data import load_data,load_validation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

start=time.clock()
batch_size = 128
nb_classes = 32
nb_epoch = 12

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between tran and test sets


model = Sequential()

model.add(Convolution2D(32, nb_conv, nb_conv,
                        border_mode='full',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))#relu
model.add(Convolution2D(32, nb_conv, nb_conv))
model.add(Activation('relu'))#relu
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(Convolution2D(128, nb_conv, nb_conv))#64
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(512))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.load_weights("./cnn_tanh.h5")
fp=open('./result/result100.txt','r+')
chars='23456789ABCDEFGHJKLMNPQRSTUVWXYZ'
for picnum in range(2000,2100):
	s=''
	print (picnum)
	for i in range(0,4):
		data= np.empty((1,1,28,28),dtype="float32")
		path='./test2/'+str(picnum)+'_'+str(i)+'.png'
		image=Image.open(path)
		r,g,b=image.split()
		image=r
		image=image.resize((28,28))
		data[0,:,:,:]=np.asarray(image,dtype="float32")
		res=model.predict_classes(data,batch_size=128,verbose=1)
		result=chars[res]
		s+=result
	fp.writelines(str(picnum)+','+''.join(s)+'\n')
	fp.flush()
end=time.clock()
print (end-start)
