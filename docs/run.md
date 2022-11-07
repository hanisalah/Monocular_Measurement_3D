# Running the Program

1. The program has two main interfaces that the user will be dealing with. <code>main.py</code> and <code>opts.py</code>
2. <code>main.py</code> is the entry point to start running the program in train mode or predict mode.
3. <code>opts.py</code> define the settings for how the program should run. To the user, the following are the parameters that need to be defined:
    1. line 8: locker_sizes -> this is a list of 3D lists specifying the dimensions of the available lockers. The intention that in predict mode, the program will select the best locker from this list based on the dimension of the object.
    2. lines 18 through 24: This is the buildup of folders. In train mode, image_dir is the directory of images, calib_dir is the directory of calibration files, label_dir is the directory of labels files, working_dir is the directory of json annotation files. In predict mode, demo is the directory of the related images and it is used along with the calib_dir for calibration file of the camera used to shoot the image for prediction.
    3. line 37: load_model -> this is the model name to be used for inference. In case you are running in training mode set load_model to an empty string.
    4. line 43: dataset -> this is the name of the dataset to be used in training. Set it to 'custom' for a 'custom' dataset or to one of the predefined entries in default_dataset_info dictionary (starting line 277) (e.g. 'RTM3D')
    5. line 78: batch_size -> this is the batch size of the dataloader used in training. This is set to 2 due to capabilities of developer's GPU. However, it was set to 32 in the original implementation. Trying to increase the number greatly affect training duration.
    6. line 301: det_cats -> this should be the same list of det_cats used while building up the custom dataset detection categories.
4. With the above set, we are ready to run the interface.ipynb cell below
