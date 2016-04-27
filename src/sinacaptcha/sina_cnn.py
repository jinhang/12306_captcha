#-*-coding:utf-8-*-
import os
from PIL import Image
import numpy as np
import split_pic


from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils

def load_data():
    map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5,'8':6,
           'A':7, 'B':8, 'C':9, 'D':10, 'E':11, 'F':12, 'G':13, 'H':14, 'K':15, 'M':16, 'N':17,
           'P':18, 'Q':19, 'R':20, 'S':21, 'U':22, 'V':23, 'W':24, 'X':25, 'Y':26, 'Z':27}

    train_data = np.empty((15000, 1, 28, 28), dtype=np.float32)
    train_label = np.empty((15000,),dtype=np.int)
    test_data = np.empty((1500, 1, 28, 28), dtype=np.float32)
    test_label = np.empty((1500,), dtype=np.int)


    train_dirs = os.listdir("/home/jinhang/sina/train_split")
    num = len(train_dirs)
    index = 0
    for i in range(num):
        if os.path.exists('/home/jinhang/sina/train_split/' + train_dirs[i]):
            os.remove('/home/jinhang/sina/train_split/' + train_dirs[i])
        imgs = os.listdir("/home/jinhang/sina/train_split/" + train_dirs[i])
        for j in range(len(imgs)):
            train_image = Image.open("/home/jinhang/sina/train_split/" + train_dirs[i] + "/" + imgs[j])
            train_arr = np.asarray(train_image, dtype=np.float32)
            train_data[index, :, :, :] = train_arr
            train_label[index] = map[train_dirs[i]]
            index += 1


    if os.path.exists('/home/jinhang/sina/test_split/'):
        os.remove('/home/jinhang/sina/test_split/')
    test_dirs = os.listdir("/home/jinhang/sina/test_split/")
    index = 0
    for i in range(len(test_dirs)):
        if os.path.exists('/home/jinhang/sina/test_split/' + test_dirs[i] ):
            os.remove('/home/jinhang/sina/test_split/' + test_dirs[i])
        imgs = os.listdir("//home/jinhang/sina/test_split/" + test_dirs[i])
        for j in range(len(imgs)):
            test_image = Image.open("/home/jinhang/sina/test_split/" + test_dirs[i] + "/" + imgs[j])
            test_arr = np.asarray(test_image, dtype=np.float32)
            test_data[index, :, :, :] = test_arr
            test_label[index] = map[test_dirs[i]]
            index += 1

    return train_data, train_label, test_data, test_label




def load_recognize_data(i):
    imgs = os.listdir("/home/jinhang/sina/sinaTest")
    imgs.sort(key= lambda x:int(x[:-4]))

    image = Image.open("/home/jinhang/sina/sinaTest/" + imgs[i])

    return imgs[i],image


def cnn():
    train_data, train_label, test_data, test_label = load_data()
    train_data = train_data / 255.0
    test_data = test_data / 255.0

    print(train_data.shape[0], 'samples')
    print(test_data.shape[0]), 'test'
    print train_label
    
    train_label = np_utils.to_categorical(train_label, 28) # divide into 28 categories
    test_label = np_utils.to_categorical(test_label, 28)

    model = Sequential()

    model.add(Convolution2D(16, 4, 4,
                            border_mode='valid',
                            input_shape=(1, 28, 28)))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 4, 4, border_mode='full'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 4, 4, border_mode='full'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(64, 4, 4, border_mode='full'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 4, 4, border_mode='full'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28))
    model.add(Activation('softmax'))


    model.load_weights('model_weights.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit(train_data, train_label, batch_size=100, nb_epoch=10, shuffle=True, verbose=1, show_accuracy=True, validation_data=(test_data, test_label))


    json_string  = model.to_json()


    open('model_architecture.json','w').write(json_string)
    model.save_weights('model_weights.h5')


def test():
    map = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5,'8':6,
           'A':7, 'B':8, 'C':9, 'D':10, 'E':11, 'F':12, 'G':13, 'H':14, 'K':15, 'M':16, 'N':17,
           'P':18, 'Q':19, 'R':20, 'S':21, 'U':22, 'V':23, 'W':24, 'X':25, 'Y':26, 'Z':27}
    #model = model_from_json(open('model_architecture.json').read())
    json_file = open('model_architecture.json','r')
    json_model = json_file.readline()
    model = model_from_json(json_model)
    model.load_weights('model_weights.h5')

    test_num = 300
    result_list = ['']*test_num

    if os.path.exists('/home/jinhang/sina/sinaTest'):
        os.remove('/home/jinhang/sina/sinaTest')

    for i in range(test_num):
        print i
        file_name, image = load_recognize_data(i)


        image = split_pic.translate_p_mode_to_single(image)
        image = split_pic.translate_image_to_single_band(image)
        image = split_pic.remove_left_right_noisy(image)
        image = split_pic.remove_margin(image)
        slide_window_array = split_pic.slide_window(image, image.size[0] / 5, 5)
        split_image_list = split_pic.split_character_by_mean(slide_window_array, image, 5)
        data = np.empty((1, 1, 28, 28), dtype="float32")

        for j in range(5):
            split_image_list[j] = split_image_list[j].resize((28,28))
            data[0,:,:,:] = np.asarray(split_image_list[j], dtype=np.float32) / 255
            result = model.predict(data)

            for k in map.keys():
                if map[k] == result.argmax():
                    result_list[i] += k
    save_result(test_num,result_list)


def save_result(test_num,result_list):
    result_fp = open("result.txt", 'w')
    for i in range(test_num):
        result_fp.writelines(str(100+i) + ":" + result_list[i] + "\n")
    answer_fp = open("/home/jinhang/sina/sinaTest100-399.txt",'r')

    correct_five = 0
    correct_four = 0
    correct_three = 0
    correct_two = 0
    correct_one = 0
    correct_zero = 0

    line = answer_fp.readline()
    for i in range(test_num):
        correct_char_num = 0;
        answer_str = line.split(':')[1]
        for j in range(5):
            if(result_list[i][j] == answer_str[j]):
                correct_char_num += 1;

        line = answer_fp.readline()
        if correct_char_num == 5:
            correct_five += 1
        elif correct_char_num == 4:
            correct_four += 1
        elif correct_char_num == 3:
            correct_three += 1
        elif correct_char_num == 2:
            correct_three += 1
        elif correct_char_num == 1:
            correct_two += 1
        elif correct_char_num == 1:
            correct_one += 1
        elif correct_char_num == 0:
            correct_zero += 1
    result_fp.writelines('correct 5:' + str(correct_five) + '\n')
    result_fp.writelines('correct 4:' + str(correct_four) + '\n')
    result_fp.writelines('correct 3:' + str(correct_three) + '\n')
    result_fp.writelines('correct 2:' + str(correct_two) + '\n')
    result_fp.writelines('correct 1:' + str(correct_one) + '\n')
    result_fp.writelines('correct 0:' + str(correct_zero) + '\n')

def main():
    cnn()

if __name__ == '__main__':
    test()