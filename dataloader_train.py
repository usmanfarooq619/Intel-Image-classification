'''
import os
import matplotlib.pyplot as plot
import cv2
from sklearn.utils import shuffle
from random import randint
import numpy as np

def data_loader(path):
    
    images = []
    labels = []

    label = 0

    for label in os.listdir(path):

        if  label == 'glacier':
            label == 0
        elif label == 'sea':
            label == 1
        elif label == 'buildings':
            label == 2
        elif label == 'forest':
            label == 3
        elif label == 'street':
            label == 4
        elif label == 'mountain':
            label == 5

        for image in os.listdir(path + label):

            image = cv2.imread(path + label + r'/' + image)
            image = cv2.resize(image, (150,150))

            images.append(image)
            labels.append(label)

    return shuffle(images,labels,random_state = 456123857) 



Images , Labels = data_loader('F:/documents/bigdata/data/seg_train/')

Train_Images = np.array(Images)
Train_Labels = np.array(Labels)
'''

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def data_loader(path):

    image_generator = ImageDataGenerator(rescale= 1./255)

    Train = image_generator.flow_from_directory(
        path,
        target_size = (150,150),
        class_mode = 'categorical'
        )

    return Train

train = data_loader('F:/documents/bigdata/data/seg_train/')