import matplotlib.pyplot as plt
import numpy as np
import random

import os
import re
import scipy.io

random.seed(135)
np.random.seed(135)

def assemble_old(split_images, T_label, Fs_label, Fn_label, block_label, num_out = 1, log_time = True):
    # Assemble targets, shuffle data, and assign training and testing sets
    # Note that I use the log_time values currently.
    # This makes time appear linearly spaced.  Perhaps helps CNN converge faster
    if log_time:
        if num_out == 1:
            targets = (np.log(T_label).T)
        elif num_out == 2:
            targets = np.vstack((np.log(T_label), Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((np.log(T_label), Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')
    else:
        if num_out == 1:
            targets = (T_label.T)
        elif num_out == 2:
            targets = np.vstack((T_label, Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((T_label, Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')

    # Obtain indicies for the items
    ind = np.arange(split_images.shape[0])
    # Shuffel the indices
    np.random.shuffle(ind)
    print('Shuffling Index Completed...')

    # Re-arrange the data using the shuffled indices
    targets = targets[ind]

    # Split train and test
    # Set to use 80% of data for training
    train_size = round(0.8*split_images[ind].shape[0])

    # Seperate Data
    train_data = split_images[ind][:train_size]
    train_labels = targets[:train_size]
    print('Training Data and Labels Completed...', train_data.shape, train_labels.shape)

    #Seperate Labels
    test_data = split_images[ind,:,:][train_size:,:,:]
    test_labels = targets[train_size:]
    print('Testing Data and Labels Completed...', test_data.shape, test_labels.shape)
    
    return (train_data, train_labels, test_data, test_labels)

def assemble_4_block(split_images, T_label, Fs_label, Fn_label, block_label, num_out = 1, log_time = True):
    # Assemble targets, shuffle data, and assign training and testing sets
    # Note that I use the log_time values currently.
    # This makes time appear linearly spaced.  Perhaps helps CNN converge faster
    # targets = np.vstack((log_T_label,Fn_label,Fs_label)).T
    if log_time:
        if num_out == 1:
            targets = (np.log(T_label).T)
        elif num_out == 2:
            targets = np.vstack((np.log(T_label),Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((np.log(T_label),Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')
    else:
        if num_out == 1:
            targets = (T_label.T)
        elif num_out == 2:
            targets = np.vstack((T_label,Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((T_label,Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')

    test_block_ind = np.min(np.where(block_label == np.max(block_label)))
    print('Test Block Index Completed...')

    train_data = split_images[:test_block_ind].clip(0)
    train_labels = targets[:test_block_ind]
    print('Training Data and Labels Completed...', train_data.shape, train_labels.shape)

    test_data = split_images[test_block_ind:].clip(0)
    test_labels = targets[test_block_ind:]
    print('Testing Data and Labels Completed...', test_data.shape, test_labels.shape)
    
    return (train_data, train_labels, test_data, test_labels)

def assemble_1_block(split_images, T_label, Fs_label, Fn_label, block_label, image_grid_size, num_out = 1, log_time = True):
    # Assemble targets, shuffle data, and assign training and testing sets
    # Note that I use the log_time values currently.
    # This makes time appear linearly spaced.  Perhaps helps CNN converge faster
    # targets = np.vstack((log_T_label,Fn_label,Fs_label)).T
    
    if image_grid_size == 1:
        return assemble_old(split_images = split_images, T_label = T_label, 
                            Fs_label = Fs_label, Fn_label = Fn_label, 
                            block_label = block_label, num_out = num_out, 
                            log_time = log_time)
    
    if log_time:
        if num_out == 1:
            targets = (np.log(T_label).T)
        elif num_out == 2:
            targets = np.vstack((np.log(T_label), Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((np.log(T_label), Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')
    else:
        if num_out == 1:
            targets = (T_label.T)
        elif num_out == 2:
            targets = np.vstack((T_label, Fs_label)).T
        elif num_out == 3:
            targets = np.vstack((T_label, Fs_label, Fn_label)).T
        else:
            print('ERROR: INCORRECT VALUE SPECIFIED FOR num_out!!!')
            
    # I want to test if the Nueral Network works well on images it has NEVER
    # seen before.  As such, I am careful to remove a specifc square from
    # ALL the images for testing.  That way we can make sure it isn't just learning
    # features of each square in the image.

    # Obtain the sub_images we split the original image into
    
    subimage_ind = np.arange(0, image_grid_size)

    # Randomnly choose a square of ALL images to remove
    # so that we can test on it later.
    np.random.shuffle(subimage_ind)
    train_size = round((.8)*subimage_ind.shape[0])

    test_index = []

    # Fill data lists
    for i in np.arange(0, (split_images.shape[0] / image_grid_size)):
        test_index.append((i * image_grid_size) + subimage_ind[train_size:])

    # Make sure index values are integers
    test_index = np.array(test_index).astype(np.intc).flatten()
    print('Test Index Completed...')

    train_data = np.delete(split_images, test_index, axis = 0)
    train_labels = np.delete(targets, test_index, axis = 0)
    print('Training Data and Labels Completed...')

    test_data = split_images[test_index]
    test_labels = targets[test_index]
    print('Testing Data and Labels Completed...')


    # TROUBLESHOOTING: Plot an array of some of the images, to try visually confirm proper selection.
    fig,ax = plt.subplots(4,8,figsize=(20,10))
    for i in range(32):
        ax[i//8,i%8].matshow(test_data[i],cmap=plt.cm.gray)
        ax[i//8,i%8].set_xticks(())
        ax[i//8,i%8].set_yticks(())
        
    return (train_data, train_labels, test_data, test_labels)

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