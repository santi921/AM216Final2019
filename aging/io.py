'''
This file contains functions used to handle data 
'''
import matplotlib.pyplot as plt
import numpy as np
from time import gmtime, strftime
import random

import os
import re
import scipy.io

random.seed(135)
np.random.seed(135)

def blockshaped(arr, size):
    """
    Breaks an array into smaller pieces.  Returns an array of shape (n, nrows, ncols) where
    n * size * size = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    
    The returned array is indexed starting in the top left of the image and moving across to the right
    until hitting the edge.  At which point it continues one line down.
    
    Inputs
    arr: 2D Array to be split
    size: Size that split 2D arrays should be
    """
    h, w = arr.shape
    return (arr.reshape(h//size, size, -1, size)
               .swapaxes(1,2)
               .reshape(-1, size, size))

def crop_center(img,crop):
    """
    Crops an image into a square from the center outwards.
    
    Inputs
    img: 2D Array to be cropped
    crop: size to crop image to
    """
    # Obtain the shape of the image
    if len(img.shape) == 3:
        x, y, c = img.shape
#         The starting coordinates of the new image
        startx = x//2 - crop//2
#         print(x//2, startx)
        starty = y//2 - crop//2    
#         print(y//2, starty)
        return img[startx : startx + crop, starty : starty + crop, :]
    else:
        x, y = img.shape
#         The starting coordinates of the new image
        startx = x//2 - crop//2
#         print(x//2, startx)
        starty = y//2 - crop//2    
#         print(y//2, starty)
        return img[startx : startx + crop, starty : starty + crop]

def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)

def file_names(direc = 'D:/MLdata/', num_samples = 1, num_exps = 200):
    """
    Obtains file names of the data we want to import.
    
    This function assumes data is in a folder of the form:
        direc + folders[i] + '/Extract Data/'
    
    Inputs
    direc: parent location of different samples
    num_samples: number of samples we want to input
    num_exps = number of TOTAL experiments we want to obtain
    """
    folders  = [filename for filename in os.listdir(direc)][:num_samples]
    files = []
    
    for i in range(len(folders)):
        if num_exps > 0:
            #Randomly selects files for each block
            temp = random.sample(os.listdir(direc + folders[i] + '/Extract Data/'), int(num_exps/len(folders)))
            files.extend(temp)   
        else:
            temp = os.listdir(direc + folders[i] + '/Extract Data/')
            files.extend(temp)
    return (folders,files)


def subtractor(Fn, Fs, T, images):
    # Make the values the final values
    Fn = [Fn[-1]]
    Fs = [Fs[-1]]
    #Take difference of time values
    T = [T[-1] - T[0]]

    # Make the image the difference between the final and initial block
    images = (images[:,:,-1] - images[:,:,0]).clip(0)
    
    return (Fn, Fs, T, images)
    
    
def data_extractor(files, folders, direc = 'D:/MLdata/', crop_size = 750, split_size = 250, clip_value = 10, subtract = False, log_image = True):
    # Intialize Lists
    length_index = []
    split_images = []
    Fn_label = []
    Fs_label = []
    T_label = []
    block_label = []
    exp_label = []

    for file in files:
        block_num = int(re.search('block(.*)_', file).group(1))
        exp_num = int(re.search('Exp(.*).mat', file).group(1))

        print(direc + folders[block_num - 1] + '/Extract Data/' + file, end="\r")
        # Import Matlab data
        data = scipy.io.loadmat(direc + folders[block_num - 1] + '/Extract Data/' + file)

        # Get data values from Matlab file
        Fn = data['Fn'].flatten()
        Fs = data['Fs'].flatten()
        T = data['T'].flatten()

        if len(T) < 10:
            continue

        images = data['images'].astype(np.float)
        
        # Take difference of images if we want to.
        if subtract:
            Fn, Fs, T, images = subtractor(Fn, Fs, T, images)

        if (len(Fn) == len(Fs) == len(T)):
            length_index.append(len(Fn))
        else:
            print('ERROR: INCONSISTENT SIZE FOR FILE:', direc + folders[block_num - 1] + '/Extract Data/' + file)
            break


        # Pull block and experiment values from filename
        block = np.repeat(block_num, len(T))
        exp = np.repeat(exp_num, len(T))

        # Crop images to appropriate size
        cropped_images = crop_center(images, crop_size)

        if len(cropped_images.shape) == 3:
            for i in range(cropped_images.shape[2]):
                #Break images into smaller pieces
                #Divide by 255 to normalize data
                if log_image:
                    temp = np.log(blockshaped(cropped_images[:,:,i], split_size).clip(clip_value))/np.log(255.0)
                else:
                    temp = blockshaped(cropped_images[:,:,i], split_size)/255.0
                
                split_images.extend(temp)
        else:
            #Break images into smaller pieces
            #Divide by 255 to normalize data
            if log_image:
                temp = np.log(blockshaped(cropped_images, split_size).clip(clip_value))/np.log(255.0)
            else:
                temp = blockshaped(cropped_images, split_size)/255.0
            
            split_images.extend(temp)

        # Append data to lists
        # Adjust labels to account for the splitting of images
        Fn_label.extend(np.repeat(Fn, (crop_size/split_size)**2))
        Fs_label.extend(np.repeat(Fs, (crop_size/split_size)**2))
        T_label.extend(np.repeat(T, (crop_size/split_size)**2))
        block_label.extend(np.repeat(block, (crop_size/split_size)**2)) 
        exp_label.extend(np.repeat(exp, (crop_size/split_size)**2))

    # Convert final results into arrays
    length_index = np.array(length_index)
    split_images = np.array(split_images)
    Fn_label = np.array(Fn_label)
    Fs_label = np.array(Fs_label)
    T_label = np.array(T_label)
    block_label = np.array(block_label) 
    exp_label = np.array(exp_label)
    
    return (length_index, split_images, Fn_label, Fs_label, T_label, block_label, exp_label)

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