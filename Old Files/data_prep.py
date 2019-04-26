import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import re

#Progress bar import
import time
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

def data_import_Jedi(csv_dir = ''):
    """
    DOC
    Function to crop images and place them in a newly created subdirectory.
    Inputs:
    block_num: (int) Number of the current block
    exp_num: (int) Number of the current experiment
    csv_dir: (str) Directory which holds the original files.
    """
    #Get all folder names
    image_folders = [filename for filename in os.listdir(csv_dir+'sub_images/')]

    image_data = []
    labels = []
    times = []
    normals = []
    shears = []
    indices = []

    for folder in tqdm(image_folders[:15]):
        # Get 3D matrix of data
        data = []
        label = []
        for filename in tqdm(os.listdir(csv_dir+'sub_images/'+folder+'/')):
            if 'image' in filename:
                #Read in data
                data.append(np.loadtxt(csv_dir+'sub_images/'+folder+'/'+filename, delimiter=','))
                #Obtain the image number
                result = re.search('image_(.*)_x', filename)
                label.append(float(result.group(1))) 
        image_data.append(data) # collect datas
        labels.append(label)
        stats = np.loadtxt(csv_dir+'sub_images/'+folder+'/labels_' + folder +'.csv', delimiter=',')

        # Append stats
        indices.append(stats[:,0])
        times.append(np.log(stats[:,1]))
        shears.append(stats[:,2])
        normals.append(stats[:,3])
    return (indices, times, shears, normals, image_data, labels)

def data_import_Extract(csv_dir = ''):
    """
    DOC
    Function to crop images and place them in a newly created subdirectory.
    Inputs:
    block_num: (int) Number of the current block
    exp_num: (int) Number of the current experiment
    csv_dir: (str) Directory which holds the original files.
    """
    #Get all folder names
    image_folders = [filename for filename in os.listdir(csv_dir+'sub_images/')]

    image_data = []
    labels = []
    times = []
    normals = []
    shears = []
    indices = []

    for folder in tqdm(image_folders[:15]):
        # Get 3D matrix of data
        data = []
        label = []
        for filename in tqdm(os.listdir(csv_dir+'sub_images/'+folder+'/')):
            if 'image' in filename:
                #Read in data
                data.append(np.loadtxt(csv_dir+'sub_images/'+folder+'/'+filename, delimiter=','))
                #Obtain the image number
                result = re.search('image_(.*)_x', filename)
                label.append(float(result.group(1))) 
        image_data.append(data) # collect datas
        labels.append(label)
        stats = np.loadtxt(csv_dir+'sub_images/'+folder+'/labels_' + folder +'.csv', delimiter=',')

        # Append stats
        indices.append(stats[:,0])
        times.append(np.log(stats[:,1]))
        shears.append(stats[:,2])
        normals.append(stats[:,3])
    return (indices, times, shears, normals, image_data, labels)