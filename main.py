import numpy as np
import os
from flax import linen as nn
from dataloader import dataloader
from plots import hartley_fourier
from preprocessing import dataprocessing
from datasplitting import splitting


# define the file path for the original data
datapath = '/home/natalia/Documentos/Dados/Tsinghua BCI Lab - Benchmark Dataset/'


"""
    Load data from from different stimulus frequency: 8, 10, 12 and 15 Hertz and
    choosing 16 electrodes (O1, O2, Oz, POz, Pz, PO3, PO4, PO7, PO8, P1, P2, Cz, C1, C2, CPz, FCz),
    following Vargas et. al (2022).

"""

sel_electrodes = {60:'O1', 61:'Oz', 62:'O2', 55:'POz', 47:'Pz', 54:'PO3', 56:'PO4', 52:'PO7',
              58:'PO8', 46:'P1', 48:'P2', 27:'Cz', 26:'C1', 28:'C2', 37:'CPZ', 18:'FCz'}
stimulif = [10,15]#[8,10,12,15]

eegdata, eeglabels = dataloader(subject=1, electrode="All", stimulus_frequency=stimulif, trial=False, path=datapath)
#print(eegdata.shape, eeglabels.shape)

"""
    Plotting examples from stimulus frequency of 15Hz and 10Hz for one subject, electrode and one random trial.
    
"""

# input15, l15 = dataloader(subject=1, electrode=61, stimulus_frequency=[15], trial=1, path=datapath)
# hartley_fourier(signal=np.reshape(input15, (1000)), stimulus_frequency=15, sampling_frequency=250)

# input10, l10 = dataloader(subject=1, electrode=61, stimulus_frequency=[10], trial=1, path=datapath)
# hartley_fourier(signal=np.reshape(input10, (1000)), stimulus_frequency=10, sampling_frequency=250)

""" 

    _Data Processing_
    
    Implements the first 3 stages of the methodology: 
    Slicing into blocks, 1D-DHT and pooling.
    
"""
processed_data, processed_labels = dataprocessing(data=eegdata, labels=eeglabels, n_levels=2, band_width=1)
#print(processed_data.shape, processed_labels.shape)

""" 

    _Data Splitting_
    
    Splits the data into train and test sets using sklearn train_test_plit function.
    Here we take care of shuffling the data and stratify according to the labels set.
    
"""

x_train, x_val, x_test, y_train, y_val, y_test = splitting(data=processed_data, labels=processed_labels, test_size=0.30, val_size=0.15, n_classes=2)
print(x_train, y_train)