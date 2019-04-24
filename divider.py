import csv
import numpy as np 
import pandas as pd
import os, os.path
import re

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
    if csv_dir:
        sub_dir = (csv_dir + 'ML_block' 
                    + str(block_num) 
                    + '_Exp'+str(exp_num))
    else:
        sub_dir = (csv_dir + 'ML_block' 
                    + str(block_num) 
                    + '_Exp'+str(exp_num))
    
    #Obtain time-values
    time  =  np.genfromtxt(sub_dir + '/Time/T_ML_block'
                + str(block_num) + '_Exp'+ str(exp_num) + '.csv', delimiter=',')

    #Obtain shear-values
    shear =  np.genfromtxt(sub_dir + '/Shear/Fs_ML_block'
                + str(block_num) + '_Exp' + str(exp_num) + '.csv', delimiter=',')

    #Obtain normal-values
    normal =  np.genfromtxt(sub_dir + '/Normal/Fn_ML_block'
                + str(block_num)+'_Exp' + str(exp_num) + '.csv', delimiter=',')
    
    #Create a directory for cropped images (if one doesn't exist)
    
    if not os.path.exists(sub_dir + "/sub_images/Cropped/"):
        os.mkdir(sub_dir + "/sub_images/Cropped/")
#         os.mkdir("./sub_images"+ "/exp" + str(exp_num) + "_block"+ str(block_num))

    stats = []
    labels = []

    #Go through all images from each experiment and crop them into a usable form
    #i starts at -1 to account for the incrementing factor in the for loop.
    i = -1
    for file in os.listdir(sub_dir + '/Contact'):
        file_dir = sub_dir + '/Contact/' + file
        #Check to make-sure the path is a file, not a directory
        if not os.path.isdir(file_dir):
            #Obtain raw-image 2-D array
            raw_image =  np.genfromtxt(file_dir, delimiter=',')
            #Only increment counter if we are looking at a file.
            i +=1

        #Extract label values 
        time_val   = time[i]
        shear_val  = shear[i]
        normal_val = normal[i]
        labels.append([i,time_val, shear_val, normal_val])
        
        data = pd.read_csv('ML_block1_Exp201/Contact/image_1_ML_block1_Exp201.csv', header = None).values

        #pd.read_csv('ML_block'+ str(block_num) +'_Exp'+ str(exp_num) +'/Contact/' + file, header = None)

        dim = ((np.asarray(data.shape)+1)/2).astype('int')
        data = data[dim[0]-750:dim[0]+750, dim[1]-750:dim[1]+750]

        for i in range(15):
            for j in range(15):
                temp = data[(i)*100:(i+1)*100, j*100:(j+1)*100]
                fname = "./sub_images"+ "/exp" + str(exp_num) + "_block" + str(block_num) + "/" + 'image' + str(ind) + 'x' + str(i) + 'y' + str(j) + ".csv"
                np.savetxt(fname, temp, delimiter=',', newline='\n',fmt='%i')

    stats_numpy = np.array(stats).reshape(-1,4)
    stats_numpy = stats_numpy[stats_numpy[:,0].argsort()]
    np.savetxt("./sub_images"+ "/exp" + str(exp_num) + "_block" + str(block_num) + "/folder_stats.csv",stats_numpy , delimiter=",")



if __name__ == "__main__":


	image_folders = [filename for filename in os.listdir('.') if filename.startswith('ML')]

	for folders in image_folders:
	    vals = re.findall(r'\d+', folders)        
	    exp_num = vals[0]
	    block_num = vals[1]

	    single_folder(exp_num,block_num)
