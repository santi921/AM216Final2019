# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:55:07 2019

@author: Nicholas
"""
import matplotlib.pyplot as plt
import numpy as np
import random

random.seed(135)
np.random.seed(135)

def random_shuffle(split_images, ML_labels, label_dic, log_time = True):
    # Assemble targets, shuffle data, and assign training and testing sets
    label_list = []
    for key in ML_labels:
        if key == 'T' and log_time:
            label_list.append(np.log(label_dic[key]))
            label_dic.pop(key, None)
        else:
            label_list.append(label_dic[key])
            label_dic.pop(key, None)
    
    targets = np.vstack(tuple(label_list)).T
    
    train_size = int(np.floor(0.8*split_images.shape[0]))
    
    # Shuffle the indices
    ind = np.arange(split_images.shape[0])
    np.random.shuffle(ind)
    
    train_ind = np.sort(ind[:train_size])
    test_ind = np.sort(ind[train_size:])
    print('Shuffling Index Completed...')

    # Seperate Data
    train_data = split_images[train_ind]
    train_labels = targets[train_ind]
    print('Training Data and Labels Completed...', train_data.shape, train_labels.shape)

    #Seperate Labels
    test_data = split_images[test_ind]
    test_labels = targets[test_ind]
    print('Testing Data and Labels Completed...', test_data.shape, test_labels.shape)
    
    train_metadata = {key:label_dic[key][train_ind] for key in label_dic}
    test_metadata = {key:label_dic[key][test_ind] for key in label_dic}
    print('Training and Testing Metadata Completed...')
    
    return (train_data, train_labels, train_metadata, test_data, test_labels, test_metadata)

def withold_exp(split_images, ML_labels, label_dic, log_time = True, norm_split = False):
    
    if norm_split:
        for i in range(split_images.shape[0]):
            split_images[i] /= np.max(split_images[i])
    
    label_list = []
    for key in ML_labels:
        if key == 'T' and log_time:
            label_list.append(np.log(label_dic[key]))
            label_dic.pop(key, None)
        else:
            label_list.append(label_dic[key])
            label_dic.pop(key, None)
    
    targets = np.vstack(tuple(label_list)).T
    blocks = label_dic['Block']
    test_block_ind = np.min(np.where(blocks == np.max(blocks)))
    print('Test Block Index Completed...')

    train_data = split_images[:test_block_ind].clip(0)
    train_labels = targets[:test_block_ind]
    print('Training Data and Labels Completed...', train_data.shape, train_labels.shape)

    test_data = split_images[test_block_ind:].clip(0)
    test_labels = targets[test_block_ind:]
    print('Testing Data and Labels Completed...', test_data.shape, test_labels.shape)
    
    train_metadata = {key:label_dic[key][:test_block_ind] for key in label_dic}
    test_metadata = {key:label_dic[key][test_block_ind:] for key in label_dic}
    print('Training and Testing Metadata Completed...')
    
    return (train_data, train_labels, train_metadata, test_data, test_labels, test_metadata)

def withhold_sqr(split_images, ML_labels, label_dic, image_grid_size, log_time = True, norm_split = False, cols = 4, rows = 4):
    # Assemble targets, shuffle data, and assign training and testing sets
    
    if image_grid_size == 1:
        return random_shuffle(split_images = split_images, ML_labels = ML_labels, 
                            block_label = label_dic['Block'], log_time = log_time)
        
    if norm_split:
        for i in range(split_images.shape[0]):
            split_images[i] /= np.max(split_images[i])
    
    label_list = []
    for key in ML_labels:
        if key == 'T' and log_time:
            label_list.append(np.log(label_dic[key]))
            label_dic.pop(key, None)
        else:
            label_list.append(label_dic[key])
            label_dic.pop(key, None)
    
    targets = np.vstack(tuple(label_list)).T
    
    # I want to test if the Nueral Network works well on images it has NEVER
    # seen before.  As such, I am careful to remove a specifc square from
    # ALL the images for testing.  That way we can make sure it isn't just learning
    # features of each square in the image.

    # Obtain the sub_images we split the original image into
    
    subimage_ind = np.arange(0, image_grid_size)

    # Randomnly choose a square of ALL images to remove
    # so that we can test on it later.
    np.random.shuffle(subimage_ind)
    train_size = int(np.floor((.8)*subimage_ind.shape[0]))

    test_index = []

    # Fill data lists
    for i in np.arange(0, (split_images.shape[0] / image_grid_size)):
        test_index.append((i * image_grid_size) + subimage_ind[train_size:])

    test_index = np.sort(np.array(test_index).astype(np.intc).flatten())
    print('Test Index Completed...')

    train_data = np.delete(split_images, test_index, axis = 0)
    train_labels = np.delete(targets, test_index, axis = 0)
    print('Training Data and Labels Completed...', train_data.shape, train_labels.shape)

    test_data = split_images[test_index]
    test_labels = targets[test_index]
    print('Testing Data and Labels Completed...', test_data.shape, test_labels.shape)
    
    dic_keys = list(label_dic.keys())
    for key in dic_keys:
        if key in ML_labels:
            try:
                dic_keys.remove(key)
            except ValueError:
                pass
    
    train_metadata = {key:np.delete(label_dic[key], test_index, axis = 0) for key in dic_keys}
    test_metadata = {key:label_dic[key][test_index] for key in dic_keys}
    print('Training and Testing Metadata Completed...')
    
    # TROUBLESHOOTING: Plot an array of some of the images, to try visually confirm proper selection.
    fig,ax = plt.subplots(rows,cols,figsize=(3*cols,4*cols))
    for i in range(rows*cols):
        ax[i//cols, i%cols].matshow(test_data[i],cmap=plt.cm.gray)
        ax[i//cols, i%cols].set_xticks(())
        ax[i//cols, i%cols].set_yticks(())
    plt.tight_layout()
    plt.show()
        
    return (train_data, train_labels, train_metadata, 
            test_data, test_labels, test_metadata)

def younger(train_data, train_labels, test_data, test_labels, time_cut = 4):
    ### REMOVE OLD ###
    train_index = np.arange(train_data.shape[0])
    test_index = np.arange(test_data.shape[0])

    train_labels, train_index = zip(*sorted(zip(train_labels, train_index)))
    test_labels, test_index = zip(*sorted(zip(test_labels, test_index)))

    train_index = np.array(train_index)
    test_index = np.array(test_index)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    nomorethan = np.where(train_labels>time_cut)[0][0]
    train_index = train_index[:nomorethan]
    train_labels = train_labels[:nomorethan]

    nomorethan = np.where(test_labels>time_cut)[0][0]
    test_index = test_index[:nomorethan]
    test_labels = test_labels[:nomorethan]

    test_data = test_data[test_index,:,:]
    train_data = train_data[train_index,:,:]
    
    return (train_data, train_index, test_data, test_index)