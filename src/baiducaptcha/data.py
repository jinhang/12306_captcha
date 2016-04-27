#coding:utf-8
from __future__ import division
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2gray
import h5py

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，图像大小28*28
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
	data= np.empty((8000,1,28,28),dtype="float32")
	label= np.empty((8000,),dtype="uint8")
	imgs= os.listdir("./train")
	num= len(imgs)
	for i in range(num):
		img= Image.open("./train/"+imgs[i])
		#print (imgs[i])
		arr= np.asarray(img,dtype="float32")
		data[i,:,:,:]= arr
		label[i] = int(imgs[i].split('_')[2])
	data=data/255
	#print data
	return data,label


def load_validation():
	data= np.empty((1593,1,28,28),dtype="float32")
	label= np.empty((1593,),dtype="uint8")
	imgs= os.listdir("./validation")
	num= len(imgs)
	for i in range(num):
		img= Image.open("./validation/"+imgs[i])
		arr= np.asarray(img,dtype="float32")
		data[i,:,:,:]= arr
		label[i] = int(imgs[i].split('_')[2])
	data=data/255
	#print "validation"
	#print data
	return data,label
