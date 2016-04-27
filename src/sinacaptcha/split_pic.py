#-*-coding:utf-8-*-
import numpy as np
import Image
import math

def translate_image_to_single_band(image):
    image2 = Image.new('L', (image.size[0], image.size[1]), 'white')
    for i in range(image.size[0]):
        for j in range(image.size[1]):
            temp_array1 = np.asarray(image.getpixel((i, j)))
            temp_array2 = np.asarray((30, 59, 11, 0))
            gray_value = sum(temp_array1 * temp_array2) / 100
            image2.putpixel((i, j), gray_value)
    return image2

def translate_p_mode_to_single(image):
    image_array = np.asarray(image)
    image_array = image_array.copy()
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            if image_array[i, j] != 0:
                image_array[i, j] = 0
            elif image_array[i, j] == 0:
                image_array[i, j] = 255
    return Image.fromarray(image_array)

def slide_window(image, window_width, stride):
    length = (image.size[0] - window_width) / stride + 1
    predict_data_array = np.empty((length, 1, image.size[1], window_width), dtype=np.float32)
    crop_left = 0
    for i in range(length):
        crop_image = image.crop((crop_left, 0, crop_left + window_width, image.size[1]))
        temp_array = np.asarray(crop_image, dtype=np.float32)  #crop_image=宽*高  np.asarray(image) = 高*宽
        predict_data_array[i, 0, :, :] = temp_array
        crop_left += stride

    return predict_data_array

def draw_image_gravity(image):
    W = 0
    X = 0
    Y = 0
    image_pix = image.load()
    for w in range(image.size[0]):
        for h in range(image.size[1]):
            if(image_pix[w,h] == 0.):
                W += 1
                X += w
                Y += h

    x = X / W
    y = Y / W
    image_pix[x,y] = 177.

def find_image_gravity(image_array):
    W = 0
    X = 0
    Y = 0
    for w in range(image_array.shape[0]):
        for h in range(image_array.shape[1]):
            if(image_array[w,h] == 0.):
                W += 1
                X += w
                Y += h

    x = X / W
    y = Y / W
    return x, y


def remove_margin(image):
    image_pix = image.load()
    left_x = 0
    left_y = 0
    flag = False
    for w in range(image.size[0]):
        if not flag:
            for h in range(image.size[1]):
                if image_pix[w, h] == 0.:
                    left_x = w
                    left_y = h
                    flag = True
                    break

    top_x = 0
    top_y = 0
    flag = False
    for h in range(image.size[1]):
        if not flag:
            for w in range(image.size[0]):
                if image_pix[w, h] == 0.:
                    top_x = w
                    top_y = h
                    flag = True
                    break

    bottom_x = 0
    bottom_y = 0
    flag = False
    for h in range(image.size[1] - 1, -1, -1):
        if not flag:
            for w in range(image.size[0] - 1, -1, -1):
                if image_pix[w, h] == 0.:
                    bottom_x = w
                    bottom_y = h
                    flag = True
                    break

    right_x = 0
    right_y = 0
    flag = False
    for w in range(image.size[0] - 1, -1, -1):
        if not flag:
            for h in range(image.size[1] - 1, -1, -1):
                if image_pix[w, h] == 0.:
                    right_x = w
                    right_y = h
                    flag = True
                    break

    return image.crop((left_x, top_y, right_x, bottom_y + 1))

def project_to_x(raw_image):
    image_load = raw_image.load()
    x_count_list = []
    for i in range(raw_image.size[0]):
        count = 0
        for j in range(raw_image.size[1]):
            if image_load[i, j] == 0.:
                count += 1
        x_count_list.append(count)

    return x_count_list

def draw_line(slide_window_array, index, raw_image):
    if index >= slide_window_array.shape[0]:
        index = slide_window_array.shape[0] - 1
    x, y = find_image_gravity(slide_window_array[index, 0, :, :])
    split_line_x = index * 4
    if x + 10 < slide_window_array.shape[3]:
        split_line_x = split_line_x + x + 10
    else:
        split_line_x = split_line_x + slide_window_array.shape[3]

    raw_image_pix = raw_image.load()
    for i in range(raw_image.size[1]):
        raw_image_pix[split_line_x, i] = 177.

    return split_line_x

def calculate_next_index(raw_image, split_x):
    return int(math.ceil(split_x / 4.))

def remove_left_right_noisy(img):
    pix = img.load()

    thresh = 4
    left = 0
    right = img.size[0]-1

    pix_list = [0]*100
    for w in range(img.size[0]):
        for h in range(img.size[1]):
            if pix[w, h] == 0:
                pix_list[w] += 1

    for i in range(len(pix_list)/3):
        if pix_list[0] > thresh:
            break
        if pix_list[i] <= thresh and pix_list[i+1] > thresh:
            left = i+1
            break
    for i in range(len(pix_list)-1, len(pix_list)/3, -1):
        if pix_list[i] <= thresh and pix_list[i - 1] > thresh:
            right = i-1
            break

    return img.crop((left, 0, right, img.size[1]))

def split_character_by_gravity(slide_window_array, raw_image, character_num):
    split_x = draw_line(slide_window_array, 0, raw_image)

    index = calculate_next_index(raw_image, split_x)
    split_x = draw_line(slide_window_array, index, raw_image)

    index = calculate_next_index(raw_image, split_x)
    split_x = draw_line(slide_window_array, index, raw_image)

def split_character_by_mean(slide_window_array, raw_image, character_num):
    raw_image_array = np.asarray(raw_image)
    split_image_list = []

    split_x = draw_line(slide_window_array, 0, raw_image)
    split_image_list.append(Image.fromarray(raw_image_array[:, 0: split_x + 4]))
    last_split_x = split_x

    for i in range(character_num - 2):
        index = calculate_next_index(raw_image, split_x)
        split_x = draw_line(slide_window_array, index, raw_image)
        split_image_list.append(Image.fromarray(raw_image_array[:, last_split_x - 3: split_x + 4]))
        last_split_x = split_x

    split_image_list.append(Image.fromarray(raw_image_array[:, last_split_x - 3: ]))

    return split_image_list




def main():
    answer_file_handler = open('/home/jinhang/sina/sinaTest100-399.txt')
    answer_file_list = answer_file_handler.readlines()
    answer_file_handler.close()

    for i in range(300):
        print i
        image = Image.open('/home/jinhang/sina/sinaTest/' + str(i+100) + '.jpg')
        image = translate_p_mode_to_single(image)
        image = translate_image_to_single_band(image)
        image = remove_left_right_noisy(image)
        image = remove_margin(image)
        slide_window_array = slide_window(image, image.size[0] / 5, 5)
        split_image_list = split_character_by_mean(slide_window_array, image, 5)
        answer = answer_file_list[i].split(':')[1][:5]

        for j in range(5):
            split_image_list[j].resize((28,28)).save('/home/jinhang/sina/test_split/' + answer[j] + '/' + str(i) + str(j) + '.gif')
            if split_image_list[j].resize((28,28)).size != (28,28):
                print split_image_list[j].resize((28,28)).size,j

    '''
    for i in range(slide_window_array.shape[0]):
        temp_image = Image.fromarray(slide_window_array[i, 0, :, :])
        draw_image_gravity(temp_image)
        temp_image.save('/jinhang/sina/slide/' + str(i) + '.gif')
    '''

if __name__ == '__main__':
    main()
