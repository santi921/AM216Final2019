import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def time_plotter(length_index, files, label_dic, split_images, image_grid_size, full = False):
    #
    # Subplots are organized in a rows x cols Grid

    for i in range(len(length_index)): #Cycle through experiments
        #Give the experiment we are lookings at
        print('--------------------------------------', files[i], '--------------------------------------')
        for j in range(length_index[i]):#Cycle through full-images in experiment
            #Determine the number of columns needed for display
            if image_grid_size > 3:
                cols = int(np.sqrt(image_grid_size))
                fig = plt.figure(1, figsize=(8,8))
            else:
                cols = image_grid_size
                fig = plt.figure(1, figsize=(10,10))

            #Determine the number of rows needed
            rows = image_grid_size // cols 
            rows += image_grid_size % cols

            sub_start = np.sum(length_index[:i]) + j
            sub_start *= image_grid_size
            # Give the time-frame in the experiment we are looking at
            print('##########', ('Fs = ' + str(round(label_dic['Fs'][sub_start],2)) + 
              ', Fn = ' + str(round(label_dic['Fn'][sub_start],2)) +
              ', T = ' + str(round(label_dic['T'][sub_start],2))), '##########')
          # add every single subplot to the figure with a for loop
            for k in range(1, image_grid_size + 1): 
                #Cycle through sub-images of image
                ax = fig.add_subplot(rows, cols, k)
                ax.matshow(split_images[sub_start + k - 1], cmap=plt.cm.gray)
                ax.set_xticks(())
                ax.set_yticks(())
            plt.tight_layout()
            plt.show()
            if not full:
                break

            
def diff_plotter(Fs_label, Fn_label, T_label, split_images, length_index, files, image_grid_size):
    # Show difference values for all split_images
    initial = 0
    diff = []
    
    for i in range(len(length_index)):
        #Give the experiment we are lookings at
        print('-------------------', files[i], '-------------------')
        final = initial + (length_index[i]-1)*image_grid_size
        print('initial:', initial, 'final:', final)
        
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, figsize = (24,24))
        
        ax = axs[0]
        ax.set_title(('Fs = ' + str(round(Fs_label[initial],2)) + 
                      ', Fn = ' + str(round(Fn_label[initial],2)) +
                      ', T = ' + str(round(T_label[initial],2))))
        ax.axis('off')
        ax.matshow(split_images[initial], cmap=plt.cm.gray)
    
        ax = axs[1]
        ax.set_title(('Fs = ' + str(round(Fs_label[final],2)) + 
                      ', Fn = ' + str(round(Fn_label[final],2)) +
                      ', T = ' + str(round(T_label[final],2))))
        ax.axis('off')
        ax.matshow(split_images[final], cmap=plt.cm.gray)
        
        ax = axs[2]
        temp = (split_images[final] - split_images[initial,:,:])
        diff.append(temp)
        ax.set_title(('Difference, $\Delta = $' + str(round(np.sum(temp),2))))
        ax.axis('off')
        ax.matshow(temp, cmap=plt.cm.gray)
        
        plt.tight_layout()
        plt.show()
        initial = final + 1
    diff = np.array(diff)
    
def intensity_plotter(length_index, label_dic, split_images, 
                      image_grid_size, normalize_to_1 = True, 
                      normalize_all = True, log_plot = False):
    
    #Obtain the intensity values of each image
    inten_data = np.sum(split_images, axis = (1,2))
    if normalize_to_1:
        inten_data /= np.max(inten_data)
    #Create lables to be used for the figures
    Fs = np.round(label_dic['Fs'], 2)
    T = np.round(label_dic['T'], 4)
    blk = label_dic['Block']
    exp = label_dic['Exp']
    #Obtain the number of experiments
    num_exp = len(length_index)
    if num_exp < 2:
        cols = num_exp
    else:
        cols = 2
    rows = int(np.ceil(num_exp / cols))
    
    for i in range(num_exp):
        
        #Create a figure for each experiment
        fig = plt.figure(1, figsize=(8,rows*4))
        ax = fig.add_subplot(rows, cols, i + 1)
        #Get the index of inten_data where this experiment begins
        exp_start_ind = np.sum(length_index[:i]) * image_grid_size
        #Get the index of inten_data where this experiment ends
        exp_end_ind = np.sum(length_index[: i + 1]) * image_grid_size

        #Add a subplot for the intensities across all sub_images
        #in an experiment.
        for k in range(1, image_grid_size + 1): 
            #Cycle through sub-images of image
            intense = inten_data[exp_start_ind + k - 1: exp_end_ind : image_grid_size]
            if normalize_all:
                intense /= np.max(intense)
            time = T[exp_start_ind + k - 1: exp_end_ind : image_grid_size]
            if log_plot:
                ax.loglogplot(time, intense, '.', label = str(k))
            else:
                ax.plot(time, intense, '.', label = str(k))
        ax.set_xlim(0,np.max(T))
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Intensity [A.U.]')
        ax.set_title('Blk: ' + str(blk[exp_start_ind]) + 
                     ' Exp: ' + str(exp[exp_start_ind]) + 
                     ' Fs: ' + str(Fs[exp_start_ind]))
        if image_grid_size < 9:
            ax.legend(loc = 'lower right')
    plt.tight_layout()
    plt.show()
    
def hist_plotter(history, savename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Validation Error')
#     plt.ylim([0,5])
    plt.legend()
    plt.savefig(savename + ".png", dpi = 600, bbox_inches='tight')
    plt.show()
    
def data_plotter(times, pred_times, savename, color = 'k', lim = 9):
    plt.figure()
    plt.scatter(times, pred_times, marker = ".", color = color, alpha = .1)
    plt.plot([0, lim], [0, lim], '-r') 
    plt.title('Data, R = '+str(np.round(pearsonr(times, pred_times)[0], 3)))
    plt.xlim(0,lim)
    plt.ylim(0,lim)
    plt.xlabel('Actual Age [ln(sec)]')
    plt.ylabel('Predicted Age [ln(sec)]')
    plt.savefig(savename + ".png", dpi = 600, bbox_inches='tight')
    plt.show()