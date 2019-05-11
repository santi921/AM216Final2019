# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:55:07 2019

@author: Nicholas
"""
import os
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from keras.utils import plot_model



def inception_train(train_data, train_labels, test_data, test_labels,
                    savedir, model, epochs = 75,
                    loss = 'mean_squared_logarithmic_error', 
                    metrics = ['mean_absolute_error', 'accuracy'],
                    patience = 25):
    #Pad data for use in inception
    sess = tf.InteractiveSession()
    
    train_data = np.tile(train_data.reshape(train_data.shape[0], train_data.shape[1],  train_data.shape[2], 1),3)
    
    test_data = np.tile(test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1),3)
    
    sess.close()
    
    optimizer = tf.keras.optimizers.RMSprop(0.0001)
    
    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    
    # print(model.summary())
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    plot_model(model, to_file=(savedir + 'incep_model.png'))
        
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor=('val_'+ metrics[0]), patience=patience)
    
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=epochs, batch_size=32, verbose=1, callbacks=[early_stop])

    
    model.save(savedir + 'incep_model.h5')
    return (model, history)

def custom_train(train_data, train_labels, test_data, test_labels,
                    savedir, model, epochs = 75,
                    loss = 'mean_squared_logarithmic_error', 
                    metrics = ['mean_absolute_error', 'accuracy'],
                    patience = 25):
    #Pad data for use in inception
    
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1],  train_data.shape[2], 1)
    
    test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], 1)
    
    optimizer = tf.keras.optimizers.RMSprop(0.0001)
    
    model.compile(loss=loss,
                optimizer=optimizer,
                metrics=metrics)
    
    # print(model.summary())
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    plot_model(model, to_file=(savedir + 'custom_model.png'))
        
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor=('val_'+ metrics[0]), patience=patience)
    
    history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=epochs, batch_size=32, verbose=1, callbacks=[early_stop])

    
    model.save(savedir + 'custom_model.h5')
    return (model, history)