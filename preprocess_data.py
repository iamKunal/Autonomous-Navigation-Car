#!/usr/bin/python2
import os
import glob
import numpy as np
import cv2
import time

img_list = glob.glob("pi/data/combined/*.png")


total_labels = 3

t = cv2.cvtColor(cv2.imread(img_list[0]), cv2.COLOR_RGB2GRAY)
dims = t.shape
inputs = dims[0]*dims[1]

label = np.identity(total_labels)


frames = np.zeros((1, inputs))
labels = np.zeros((1, total_labels))


for img in img_list:
	t = cv2.cvtColor(cv2.imread(img), cv2.COLOR_RGB2GRAY).astype(np.int32)
	div = 2 if total_labels==3 else 1
	frames = np.vstack((frames, t.reshape((1, inputs)).astype(np.float32)))
	labels = np.vstack((labels, label[int(img.split('.')[-2], 10) / div]))

	print img, label[int(img.split('.')[-2], 10) / div]


np.savez('train_data/' + str(int(time.time())) + '.npz', data=frames, labels=labels)

