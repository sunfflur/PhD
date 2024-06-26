import numpy as np
import os
from flax import linen as nn
from main_functions.dataloader import mnist_dataloader
from main_functions.preprocessing import mnist_preprocessing
from main_functions.datasplitting import splitting_bcic2a


def get_data(datapath, electrodes, classes, validation_set=True, **kwargs):

    data_train, data_test, labels_train, labels_test = mnist_dataloader(datapath, electrodes, classes)

    """ 
        _Data Processing_
        Implements the first 3 stages of the methodology: 
        Slicing into blocks, 1D-DHT and pooling.
    """
    
    processed_train = mnist_preprocessing(
        data=data_train,
        sampling_frequency=kwargs.get("sampling_frequency", 200),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 2),
        overlap=kwargs.get("overlap", 0)
    )
    
    processed_test = mnist_preprocessing(
        data=data_test,
        sampling_frequency=kwargs.get("sampling_frequency", 200),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 2),
        overlap=kwargs.get("overlap", 0)
    )
    

    """ 
        _Data Splitting_
        Splits the data into train and test sets using sklearn train_test_plit function.
        Here we take care of shuffling the data and stratify according to the labels set.
    """
    
    if validation_set == True:
        x_train, x_val, y_train, y_val = splitting_bcic2a(
            data=processed_train,
            labels=labels_train,
            n_trials=kwargs.get("n_trials", 10),
            val_trial=kwargs.get("val_trial", 0)
        )
        return x_train, x_val, processed_test, y_train, y_val, labels_test
    else:
        return processed_train, processed_test, labels_train, labels_test
