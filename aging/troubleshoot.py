import matplotlib.pyplot as plt
import numpy as np

def time_plotter(length_index, files, Fs_label, Fn_label, T_label, split_images, crop_size, split_size):
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
            print('##########', ('Fs = ' + str(round(Fs_label[arr_ind],2)) + 
              ', Fn = ' + str(round(Fn_label[arr_ind],2)) +
              ', T = ' + str(round(T_label[arr_ind],2))), '##########')
          # add every single subplot to the figure with a for loop
            for k in range(1, image_grid_size + 1): #Cycle through sub-images of image
                ax = fig.add_subplot(rows, cols, k)
                ax.matshow(split_images[arr_ind + k -1,:,:], cmap=plt.cm.gray)
                ax.set_xticks(())
                ax.set_yticks(())
            plt.tight_layout()
            plt.show()