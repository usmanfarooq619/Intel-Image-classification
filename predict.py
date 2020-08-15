import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

def data_loader(path):

    image_generator = ImageDataGenerator(rescale= 1./255)

    Pred = image_generator.flow_from_directory(
        path,
        target_size = (150,150),
        class_mode = 'categorical'
        )

    return Pred

pred = data_loader('F:/documents/bigdata/data/seg_pred/')