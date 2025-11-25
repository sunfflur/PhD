import numpy as np
import os
from main_functions.dataloader import MotionSense_dataloader
from main_functions.preprocessing import mnist_preprocessing

def get_data(datapath, **kwargs):

    train_attributes_array, train_labels, val_attributes_array, val_labels, test_attributes_array, test_labels = MotionSense_dataloader(path=datapath)

    """ 
    _Data Processing_
    Implements the first 3 stages of the methodology: 
    Slicing into blocks, 1D-DHT and pooling.
    """

    processed_train = mnist_preprocessing(
        data=train_attributes_array,
        sampling_frequency=kwargs.get("sampling_frequency", 50),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 5),
        overlap=kwargs.get("overlap", 0),
        pooling_type=kwargs.get("pooling_type", "Sum")        
    )

    processed_val = mnist_preprocessing(
    data=val_attributes_array,
    sampling_frequency=kwargs.get("sampling_frequency", 50),
    n_levels=kwargs.get("n_levels", 3),
    band_width=kwargs.get("band_width", 1),
    transform=kwargs.get("transform", "DHT"),
    window=kwargs.get("window", 5),
    overlap=kwargs.get("overlap", 0),
    pooling_type=kwargs.get("pooling_type", "Sum")        
    )

    processed_test = mnist_preprocessing(
    data=test_attributes_array,
    sampling_frequency=kwargs.get("sampling_frequency", 50),
    n_levels=kwargs.get("n_levels", 3),
    band_width=kwargs.get("band_width", 1),
    transform=kwargs.get("transform", "DHT"),
    window=kwargs.get("window", 5),
    overlap=kwargs.get("overlap", 0),
    pooling_type=kwargs.get("pooling_type", "Sum")        
    )

    return processed_train, processed_val, processed_test, train_labels, val_labels, test_labels