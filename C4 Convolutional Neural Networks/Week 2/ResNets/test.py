import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras.layers as tfl

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation

def mo():
    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)
    directory = 'C:/Users/PU0900028/Desktop/AI/C4 Convolutional Neural Networks/Week 2/Transfer Learning with MobileNet/dataset/'
    train_dataset = image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='training',
                                                seed=42)
    validation_dataset = image_dataset_from_directory(directory,
                                                shuffle=True,
                                                batch_size=BATCH_SIZE,
                                                image_size=IMG_SIZE,
                                                validation_split=0.2,
                                                subset='validation',
                                                seed=42)

  
    return None
    

mo()