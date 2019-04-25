import csv
import numpy as np 
import os, os.path
import re
from joblib import Parallel, delayed 

def single_folder(block_num, exp_num, csv_dir = ''):
    """
    DOC
    Function to crop images and place them in a newly created subdirectory.
    Inputs:
    block_num: (int) Number of the current block
    exp_num: (int) Number of the current experiment
    csv_dir: (str) Directory which holds the original files.
    """
    #Define a shorthand directory
    sub_dir = (csv_dir + 'ML_block' 
                    + str(block_num) 
                    + '_Exp'+str(exp_num))
    
    exp_dir = 'ML_block' + str(block_num) + '_Exp'+ str(exp_num) 
        
    #Obtain time-values
    time  =  np.genfromtxt(os.path.join(sub_dir, 'Time', 'T_' + exp_dir + '.csv')
                           , delimiter=',')

    #Obtain shear-values
    shear =  np.genfromtxt(os.path.join(sub_dir, 'Shear', 'Fs_' + exp_dir + '.csv')
                           , delimiter=',')

    #Obtain normal-values
    normal =  np.genfromtxt(os.path.join(sub_dir, 'Normal', 'Fn_' + exp_dir + '.csv')
                            , delimiter=',')
    
    #Create a directory for cropped images (if one doesn't exist)
    if not os.path.exists(os.path.join(csv_dir, 'sub_images', exp_dir)):
        os.makedirs(os.path.join(csv_dir, 'sub_images', exp_dir))
    
    stats = []
    
    onlyfiles = [f for f in os.listdir(sub_dir +'/Contact') 
                 if os.path.isfile(os.path.join(sub_dir +'/Contact', f))]
    
    #Go through all images from each experiment and crop them into a usable form
    for file in onlyfiles:
        #Obtain index value from current file
        #Changes image index to start at 0, instead of 1
        print(file)
        ind = int(re.findall(r'\d+', file)[0]) - 1

        #Obtain values from array
        time_val = time[ind]
        shear_val = shear[ind]
        normal_val = normal[ind]
        stats.append([ind, time_val, shear_val, normal_val])
        
        #Obtain contact data
        contact_data = np.genfromtxt(os.path.join(sub_dir, 'Contact', file)
                                     , delimiter=',')
        
        #Obtain shape of images for cropping
        dim = ((np.asarray(contact_data.shape)+1)/2).astype('int')
        contact_data = contact_data[dim[0]-750:dim[0]+750, dim[1]-750:dim[1]+750]
        
        #Make subimages
        for i in range(15):
            for j in range(15):
                temp = contact_data[(i)*100:(i+1)*100, j*100:(j+1)*100]
                fname = os.path.join(csv_dir, 'sub_images', exp_dir, 
                                     'image_' + str(ind) + '_x_' + str(i) + '_y_' + str(j) 
                                     + '_' + exp_dir + ".csv")
                np.savetxt(fname, temp, delimiter=',', newline='\n',fmt='%i')
                
    stats_numpy = np.array(stats).reshape(-1,4)
    stats_numpy = stats_numpy[stats_numpy[:,0].argsort()]
    #Make label file
    np.savetxt(os.path.join(csv_dir, 'sub_images', exp_dir, 'labels_' + exp_dir + '.csv')
               , stats_numpy, delimiter=",")
    
if __name__ == "__main__":
    
    csv_dir = 'D:/MLdata/18-01-09d-Exp/Jedi Data/CSV/'
    # csv_dir = '.'

    image_folders = [filename for filename in os.listdir(csv_dir) if filename.startswith('ML')]

    Parallel(n_jobs=-1)(
        delayed(single_folder)(re.findall(r'\d+', folders)[0], re.findall(r'\d+', folders)[1], csv_dir) 
        for folders in image_folders)

#     for folders in image_folders:
#         vals = re.findall(r'\d+', folders)        
#         exp_num = vals[1]
#         block_num = vals[0]
#         single_folder(block_num, exp_num, csv_dir)