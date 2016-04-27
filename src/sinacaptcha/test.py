#-*-coding:utf-8-*-
import os
from PIL import Image
import numpy as np


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils


def sina_load_data():
    train_set_directory = os.listdir('/home/jinhang/sina/split')
    train_data_set = np.empty((11500, 1, 28, 28), dtype=np.float32)
    train_label_set = np.empty((11500,), dtype=np.int8)
    count = 0
    for i in range(1, len(train_set_directory)):
        if os.path.exists('/home/jinhang/sina/split/' + train_set_directory[i] + '/.DS_Store'):
            os.remove('/home/jinhang/sina/split/' + train_set_directory[i] + '/.DS_Store')
        temp_subset_directory = os.listdir('/home/jinhang/sina/split/' + train_set_directory[i])
        for j in range(len(temp_subset_directory)):
            temp_image = Image.open('/home/jinhang/sina/split/' + train_set_directory[i] + '/' + temp_subset_directory[j])
            train_data_set[count, 0, :, :] = np.asarray(temp_image)
            train_label_set[count, ] = sina_word_to_digit_label(train_set_directory[i])
            count += 1



    return train_data_set, train_label_set

def cnn():
    data, label = load_data()

    print(data.shape[0], 'samples')
    label = np_utils.to_categorical(label, 10)

    model = Sequential()
    model.add(Convolution2D(4, 5, 5, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('tanh'))


    model.add(Convolution2D(8, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('tanh'))


    model.add(Dense(10))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.05, momentum=0.9, nesterov='True')
    model.compile(loss='categorical_crossentropy', optimizer=sgd, class_mode="categorical")

    model.fit(data, label, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, show_accuracy=True, validation_split=0.2)

    json_string  = model.to_json()
    #yaml_string  = model.to_yaml()

    open('model_architecture.json','w').write(json_string)
    model.save_weights('model_weights.h5')


def test():
    #model = model_from_json(open('model.json').read())
    model = model_from_json(open('model.json').readline())
    model.load_weights('model_weights.h5')
    img = Image.open("/home/jinhang/sina/mnist/5.123.jpg")
    data = np.empty((1, 1, 28, 28), dtype="float32")
    data[0,:,:,:] = np.asarray(img, dtype="float32")

    result = model.predict(data)

    print result

def main():
    load_data()

if __name__ == '__main__':
    main()