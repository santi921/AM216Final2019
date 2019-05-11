import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def time_plotter(length_index, files, label_dic, split_images, crop_size, split_size, full = False):
    # Subplots are organized in a rows x cols Grid
    image_grid_size = int(crop_size**2 / split_size**2)

    for i in range(len(length_index)): #Cycle through experiments
        #Give the experiment we are lookings at
        print('--------------------------------------', files[i], '--------------------------------------')
        for j in range(length_index[i]):#Cycle through full-images in experiment
            #Determine the number of columns needed for display
            if image_grid_size > 3:
                cols = crop_size/split_size
                fig = plt.figure(1, figsize=(8,8))
            else:
                cols = image_grid_size
                fig = plt.figure(1, figsize=(10,10))

            #Determine the number of rows needed
            rows = image_grid_size // cols 
            rows += image_grid_size % cols

            arr_ind = np.sum(length_index[:i]) + j*image_grid_size
            # Give the time-frame in the experiment we are looking at
            print('##########', ('Fs = ' + str(round(label_dic['Fs'][arr_ind],2)) + 
              ', Fn = ' + str(round(label_dic['Fn'][arr_ind],2)) +
              ', T = ' + str(round(label_dic['T'][arr_ind],2))), '##########')
          # add every single subplot to the figure with a for loop
            for k in range(1, image_grid_size + 1): #Cycle through sub-images of image
                ax = fig.add_subplot(rows, cols, k)
                ax.matshow(split_images[arr_ind*image_grid_size + k - 1], cmap=plt.cm.gray)
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