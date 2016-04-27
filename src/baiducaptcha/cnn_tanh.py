from __future__ import absolute_import
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility


from keras.callbacks import ModelCheckpoint 
from data import load_data,load_validation
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils



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
data,label = load_data()
label= np_utils.to_categorical(label, nb_classes)

v_data,v_label = load_validation()
v_label= np_utils.to_categorical(v_label, nb_classes)


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
checkpointer = ModelCheckpoint(filepath="./weight/cnn_tanh.h5", verbose=1, save_best_only=True)
model.fit(data, label, batch_size=batch_size, nb_epoch=20,shuffle=True,verbose=1,show_accuracy=True,validation_data=(v_data,v_label),callbacks=[checkpointer])
