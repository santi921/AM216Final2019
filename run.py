# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:54:43 2019

@author: Nicholas
"""
#%%############################# IMPORT ########################################

from __future__ import absolute_import, division, print_function

import numpy as np
from time import gmtime, strftime
import random


# TensorFlow and tf.keras
import tensorflow as tf

import aging as age

tf.test.gpu_device_name()

# Define directory with matlab files
# direc = 'D:/MLdata/18-01-09d-Exp/Extract Data/'
# direc  = 'D:/Nicholas_ML/18-01-09d-Exp/Extract Data/'
direc = 'D:/MLdata/'
# Define directory to save model and plots
savedir = 'D:/MLdata/Model/' + strftime("%Y-%m-%d %H-%M", gmtime()) + '/'

random.seed(135)
np.random.seed(135)

#%%############################# DATA ##########################################

# Optimized Data Extractor
crop_size = 750
split_size = crop_size // 3
clip_value = 1
image_grid_size = int(crop_size**2 / split_size**2)

file_num = 4

folders, files = age.file_names(direc = 'D:/MLdata/', 
                                num_samples = 4, 
                                num_exps = file_num)

length_index, split_images, label_dic = age.data_extractor(files, 
                     folders, 
                     direc = 'D:/MLdata/', 
                     crop_size = crop_size, 
                     split_size = split_size,
                     clip_value = clip_value,
                     subtract = True, 
                     log_image = True)

#%%############################# VISUALIZE #####################################


age.time_plotter(length_index, files, Fs_label, Fn_label, 
# T_label, split_images, crop_size, split_size)
# age.difference(Fs_label, Fn_label, T_label, split_images, 
# length_index, files, image_grid_size)

#%%############################# ASSEMBLE ######################################

train_data, train_labels, test_data, test_labels = \
        age.assemble_1_block(
            split_images = split_images, 
            label_dic = {'T':label_dic['T']},
            block_label = label_dic['Block'],
            image_grid_size = image_grid_size, 
            log_time = True)

# train_data, train_labels, test_data, test_labels = \
# age.younger(train_data, train_labels, test_data, test_labels, time_cut = 5)

# split_images = []
# T_label = []
# Fs_label = []
# Fn_label = []
# block_label = []

min_val = np.min([np.min(train_data), np.min(test_data)])

#Final Renormilization
train_data = (train_data - min_val)/(1 - min_val)
test_data = (test_data - min_val)/(1 - min_val)

#%%############################# BUILD #########################################

image_size = train_data.shape[1]
    
# Create the base model
# https://keras.io/applications/#inceptionv3
base_model = tf.keras.applications.inception_v3.InceptionV3(
                include_top=False, 
                weights='imagenet', 
                input_shape=(image_size,image_size,3))

base_model.trainable = False
    
incep_model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Conv2D(16, (3, 3)),
    tf.keras.layers.LeakyReLU(alpha=0.5),
    tf.keras.layers.Dropout(rate = 0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32),
    tf.keras.layers.LeakyReLU(alpha=0.5),
    tf.keras.layers.Dropout(rate = 0.3),
    tf.keras.layers.Dense(1),
    tf.keras.layers.LeakyReLU(alpha=0.5)
    ])

#custom_model = tf.keras.Sequential([
#        tf.keras.layers.Conv2D(8, kernel_size=(2, 2),
#                       input_shape=(image_size, image_size, 1)),
#        tf.keras.layers.LeakyReLU(alpha=0.5),
#        tf.keras.layers.Dropout(rate = 0.3),
#        tf.keras.layers.Conv2D(16, (3, 3)),
#        tf.keras.layers.LeakyReLU(alpha=0.5),
#        tf.keras.layers.Dropout(rate = 0.3),
#        tf.keras.layers.Flatten(),
#        tf.keras.layers.Dense(32),
#        tf.keras.layers.LeakyReLU(alpha=0.5),
#        tf.keras.layers.Dropout(rate = 0.3),
#        tf.keras.layers.Dense(1),
#        tf.keras.layers.LeakyReLU(alpha=0.5)

#%%############################# TRAIN #########################################

model, history = age.inception_train(
                    train_data, 
                    train_labels, 
                    test_data, 
                    test_labels,
                    savedir, 
                    incep_model)

# model, history = age.aion(train_data, train_labels, test_data, test_labels,
#                     savedir, full_model)

#%%############################# PREDICT #########################################

test_predic = model.predict(np.tile(
    test_data.reshape(test_data.shape[0], 
    test_data.shape[1], 
    test_data.shape[2], 
    1), 
        3))
train_predic = model.predict(np.tile(train_data.reshape(
    train_data.shape[0], 
    train_data.shape[1],  
    train_data.shape[2], 
    1),
        3))

#%%############################# RESULTS #######################################

age.hist_plotter(history, savedir)

train_times = train_labels.flatten()
train_pred_times = train_predic.flatten()

# train_shear = train_labels[:,1]
# train_pred_shear = train_predic[:,1]

test_times = test_labels.flatten()
test_pred_times = test_predic.flatten()

# test_shear = test_labels[:,1]
# test_pred_shear = test_predic[:,1]

age.data_plotter(train_times, train_pred_times, savedir + "Incep_Age_train" , 'b')

age.data_plotter(test_times, test_pred_times, savedir + "Incep_Age_test" , 'k')