
import csv
import numpy as np 
import cv2
import fnmatch
import os, os.path
import re

def single_folder(block_num, exp_num):
    #extract vector info

    time_vector  =  np.genfromtxt('ML_block'+str(block_num)+'_Exp'+str(exp_num)+'/Time/T_ML_block'+str(block_num)+'_Exp'+ str(exp_num) + '.csv', delimiter=',')
    shear_vector =  np.genfromtxt('ML_block'+str(block_num)+'_Exp'+str(exp_num)+'/Shear/Fs_ML_block'+str(block_num)+'_Exp'+ str(exp_num) + '.csv', delimiter=',')
    normal_vector=  np.genfromtxt('ML_block'+str(block_num)+'_Exp'+str(exp_num)+'/Normal/Fn_ML_block'+str(block_num)+'_Exp'+ str(exp_num) + '.csv', delimiter=',')
    
    os.mkdir("./sub_images"+ "/exp" + str(exp_num) + "_block"+ str(block_num))
    
    stats = []
    
    for ind, file in enumerate(os.listdir('ML_block'+ str(block_num)+'_Exp'+ str(exp_num) +'/Contact')):
        
        raw_data = open('ML_block'+ str(block_num) +'_Exp'+ str(exp_num) +'/Contact/' + file, 'rt')
        
        ind = int(re.findall(r'\d+', file)[0])-1
        temp_time   = time_vector[ind]
        temp_shear  = shear_vector[ind]
        temp_normal = normal_vector[ind]
        stats.append([ind,temp_time, temp_shear, temp_normal])
        
        
        reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
        x = list(reader)
        
    
        data = np.array(x).astype('int')
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
