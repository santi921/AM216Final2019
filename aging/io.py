# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:55:07 2019

@author: Nicholas
"""
import numpy as np
import random

import os
import re
import scipy.io

import warnings

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
    try:
        h, w = arr.shape
    except:
        raise Exception('The inputted array was not shapped for splitting. The array shape was: {}'.format(np.array(arr).shape))
    
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

def rand_file_names(direc = 'D:/MLdata/', num_samples = 1, num_exps = 200):
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

def seq_file_names(direc = 'D:/MLdata/', which_samples = [0], num_exps = 200):
    """
    Obtains file names of the data we want to import.
    DOES NOT RANDOMLY CHOOSE FILES.
    
    This function assumes data is in a folder of the form:
        direc + folders[i] + '/Extract Data/'
    
    Inputs
    direc: parent location of different samples
    num_samples: number of samples we want to input
    num_exps = number of TOTAL experiments we want to obtain
    """
    folders  = [filename for filename in os.listdir(direc)][:4]
    folders = [folders[i] for i in which_samples]
    print(folders)
    files = []
    
    for i in range(len(folders)):
        if num_exps > 0:
            temp = os.listdir(direc + folders[i] + '/Extract Data/')[:int(num_exps/len(folders))]
            files.extend(temp)   
        else:
            temp = os.listdir(direc + folders[i] + '/Extract Data/')
            files.extend(temp)
    return (folders,files)

def subtractor(Fn, Fs, T, images):
    #Take difference of time values
    T = T - T[0]

    # Make the image the difference between the final and initial block
    for i in range(1, images.shape[2]):
        images[:,:,i] = (images[:,:,i] - images[:,:,0]).clip(0)
    
    T = T[1:]
    Fn = Fn[1:]
    Fs = Fs[1:]
    images = images[:,:,1:]
    return (Fn, Fs, T, images)
    
    
def data_extractor(files, folders, direc = 'D:/MLdata/', crop_size = 750, split_size = 250, clip_value = 10, subtract = False, log_image = True, younger = 0):
    # Intialize Lists
    length_index = []
    split_images = []
    Fn_label = []
    Fs_label = []
    T_label = []
    block_label = []
    exp_label = []
    
    initial_blk = int(re.search('block(.*)_', files[0]).group(1))

    for file in files:
        block_num = int(re.search('block(.*)_', file).group(1))
        exp_num = int(re.search('Exp(.*).mat', file).group(1))

        print(direc + folders[block_num - initial_blk] + '/Extract Data/' + file, end="\r")
        # Import Matlab data
        data = scipy.io.loadmat(direc + folders[block_num - initial_blk] + '/Extract Data/' + file)

        # Get data values from Matlab file
        Fn = data['Fn'].flatten()
        Fs = data['Fs'].flatten()
        T = data['T'].flatten()

        if len(T) < 10:
            warnings.warn("File {} skipped due to small size.".format(file))
            continue

        images = data['images'].astype(np.float)
        
        if younger > 0:
            nomorethan = np.where(T>younger)[0][0]
            Fn = Fn[:nomorethan]
            Fs = Fs[:nomorethan]
            T = T[:nomorethan]
            images = images[:, :, :nomorethan]
        
        if images.shape[0] < crop_size or images.shape[1] < crop_size:
            raise Exception('Crop_size should not exceed the size of the sample. The sample size was: {}'.format(images.shape[:2]))
        
        # Take difference of images if we want to.
        if subtract:
            Fn, Fs, T, images = subtractor(Fn, Fs, T, images)

        if (len(Fn) == len(Fs) == len(T)):
            length_index.append(len(Fn))
        else:
            print('ERROR: INCONSISTENT SIZE FOR FILE:', direc + folders[block_num - initial_blk] + '/Extract Data/' + file)
            print('Fn:', len(Fn), 'Fs:', len(Fs), 'T:', len(T))
            break

        # Crop images to appropriate size
        cropped_images = crop_center(images, crop_size)

        # Pull block and experiment values from filename
        block = np.repeat(block_num, len(T))
        exp = np.repeat(exp_num, len(T))
        
        if len(cropped_images.shape) == 3:
            for i in range(cropped_images.shape[2]):
                #Break images into smaller pieces
                #Divide by 255 to normalize data
                
                if log_image:
                    temp = np.log(blockshaped(cropped_images[:,:,i], split_size).clip(clip_value))/np.log(255.0)
                else:
                    temp = blockshaped(cropped_images[:,:,i], split_size).clip(clip_value)/255.0
                
                split_images.extend(temp)
        else:
            #Break images into smaller pieces
            #Divide by 255 to normalize data
            if log_image:
                temp = np.log(blockshaped(cropped_images, split_size).clip(clip_value))/np.log(255.0)
            else:
                temp = blockshaped(cropped_images, split_size).clip(clip_value)/255.0
            
            split_images.extend(temp)
        # Append data to lists
        # Adjust labels to account for the splitting of images
        Fn_label.extend(np.repeat(Fn, (crop_size/split_size)**2))
        Fs_label.extend(np.repeat(Fs, (crop_size/split_size)**2))
        T_label.extend(np.repeat(T, (crop_size/split_size)**2))
        block_label.extend(np.repeat(block, (crop_size/split_size)**2)) 
        exp_label.extend(np.repeat(exp, (crop_size/split_size)**2))

    # Convert final results into arrays
    label_dic = {'T':np.array(T_label), 'Fn':np.array(Fn_label),
                 'Fs':np.array(Fs_label), 'Block': np.array(block_label),
                 'Exp': np.array(exp_label)}
    length_index = np.array(length_index)
    split_images = np.array(split_images)
    
    return (length_index, split_images, label_dic)