import numpy as np
import os
from flax import linen as nn
from main_functions.dataloader import dataloader
from main_functions.plots import hartley_fourier
from main_functions.preprocessing import dataprocessing
from main_functions.datasplitting import splitting, splitting_per_trial


def get_data(datapath, sel_electrodes, stimulif, subjects, validation_set=True, **kwargs):

    #for s in subjects:
    #print("s:", s)
    eegdata, eeglabels = dataloader(
        subject=subjects,
        electrode=sel_electrodes,
        stimulus_frequency=stimulif,
        trial=False,
        sec_off=kwargs.get("sec_off", 1),
        path=datapath,
    )
    # print(eegdata.shape, eeglabels.shape)

    """
        Plotting examples from stimulus frequency of 15Hz and 10Hz for one subject, electrode and one random trial.
    """

    #input15, l15 = dataloader(subject=1, electrode=61, stimulus_frequency=[15], trial=1, path=datapath)
    #hartley_fourier(signal=np.reshape(input15, (1000)), stimulus_frequency=15, sampling_frequency=250)

    #input10, l10 = dataloader(subject=1, electrode=61, stimulus_frequency=[10], trial=1, path=datapath)
    #hartley_fourier(signal=np.reshape(input10, (1000)), stimulus_frequency=10, sampling_frequency=250)

    """ 
        _Data Processing_
        Implements the first 3 stages of the methodology: 
        Slicing into blocks, 1D-DHT and pooling.
    """
    processed_data, processed_labels, ntrials = dataprocessing(
        data=eegdata,
        sampling_frequency=kwargs.get("sampling_frequency", 250),
        n_levels=kwargs.get("n_levels", 3),
        band_width=kwargs.get("band_width", 1),
        transform=kwargs.get("transform", "DHT"),
        window=kwargs.get("window", 2),
        overlap=kwargs.get("overlap", 0),
        pooling_type=kwargs.get("pooling_type", "Sum")        
    )

    """ 
        _Data Splitting_
        Splits the data into train and test sets using sklearn train_test_plit function.
        Here we take care of shuffling the data and stratify according to the labels set.
    """
    
    if validation_set == True:
        
        x_train, x_val, x_test, y_train, y_val, y_test = splitting_per_trial(
            data=processed_data.reshape(processed_data.shape[0],-1), 
            labels=processed_labels, 
            split_train=kwargs.get("split_train", [0,1,3,5]),
            split_val=kwargs.get("split_val", [2]),
            split_test=kwargs.get("split_test", [4]),
            trial_gab=ntrials,
            n_classes=kwargs.get("n_classes", 6))
    
        return x_train, x_val, x_test, y_train, y_val, y_test
    
    else:
        x_train, x_test, y_train, y_test = splitting_per_trial(
            data=processed_data.reshape(processed_data.shape[0],-1), 
            labels=processed_labels, 
            split_train=kwargs.get("split_train", [0,1,2,3,5]),
            split_val=kwargs.get("split_val", []),
            split_test=kwargs.get("split_test", [4]),
            trial_gab=ntrials,
            n_classes=kwargs.get("n_classes", 6))
    
        return x_train, x_test, y_train, y_test
