'''
ref https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial/blob/master/code/create_lmdb.py
Author :Adil Moujahid
Revised by:Qiaomu Shen
'''

import os
import glob
import random
import numpy as np
import cv2
from caffe.proto import caffe_pb2
import lmdb

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)
    return img

def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())

# Revised according to the your all project
# Removed before you create new lmdb
train_lmdb = '../data/train_lmdb'
validation_lmdb = '../data/validation_lmdb'

train_data = [img for img in glob.glob("../data/train/*jpg")]
test_data = [img for img in glob.glob("../data/test/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print 'Start Creating lmdb'

# Max size of the db, must be defined
size = 4 * 1024 * 1024 * 1024 #4 G
number = 0

in_db = lmdb.open(train_lmdb, map_size=int(size))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'cat' in img_path:
            label = 0
        else:
            label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        if number % 100 == 0:
            print number,'records have been parsed'
        number += 1
in_db.close()


in_db = lmdb.open(validation_lmdb, map_size=5 * 1024 * 1024 * 1024)
number = 0
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):

        img = cv2.imread(img_path, cv2.COLOR_BGR2GRAY)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'cat' in img_path:
            label = 0
        else:
            label = 1
        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        if number % 100 == 0:
            print number,'records have been parsed'
        number += 1
in_db.close()