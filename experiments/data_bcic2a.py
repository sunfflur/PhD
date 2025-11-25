import numpy as np
import os
from flax import linen as nn
from main_functions.dataloader import BCIC_dataloader
from main_functions.plots import hartley_fourier
from main_functions.preprocessing import mnist_preprocessing
from main_functions.datasplitting import splitting, splitting_per_trial, splitting_bcic2a


def get_data(datapath, subjects, validation_set=True, **kwargs):

    eegdatatrain, eegdatatest, eeglabelstrain, eeglabelstest = BCIC_dataloader(
        subject=subjects,
        trial_start_offset_seconds=kwargs.get("trial_start_offset_seconds", -0.5), 
        trial_stop_offset_seconds=kwargs.get("trial_stop_offset_seconds", 0.5),
        path=datapath,
    )


    """ 
        _Data Processing_
        Implements the first 3 stages of the methodology: 
        Slicing into blocks, 1D-DHT and pooling.
    """
    processed_train = mnist_preprocessing(
        data=eegdatatrain,
        sampling_frequency=kwargs.get("sampling_frequency", 250),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 5),
        overlap=kwargs.get("overlap", 0),
        pooling_type=kwargs.get("pooling_type", "Sum")        
    )
    
    processed_test = mnist_preprocessing(
        data=eegdatatest,
        sampling_frequency=kwargs.get("sampling_frequency", 250),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 5),
        overlap=kwargs.get("overlap", 0),
        pooling_type=kwargs.get("pooling_type", "Sum")      
    )

    """ 
        _Data Splitting_
        Splits the data into train and test sets using sklearn train_test_plit function.
        Here we take care of shuffling the data and stratify according to the labels set.
    """
    
    if validation_set == True:
        x_train, x_val, y_train, y_val = splitting_bcic2a(
            data=processed_train,
            labels=eeglabelstrain,
            n_trials=kwargs.get("n_trials", 6),
            val_trial=kwargs.get("val_trial", 0)
        )
        return x_train, x_val, processed_test, y_train, y_val, eeglabelstest
    
    else:
        return processed_train, processed_test, eeglabelstrain, eeglabelstest
