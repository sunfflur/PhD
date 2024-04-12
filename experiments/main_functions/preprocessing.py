import jax.numpy as jnp
from main_functions.slicing import dataslicing
from main_functions.pooling import datapooling
from main_functions.DHT import dataDHT
from main_functions.DFT import dataDFT
from main_functions.utils import NormalizeData
from sklearn.preprocessing import LabelEncoder

def dataprocessing(data, labels, n_levels: int, band_width: int, transform: str):
    eegdata_sliced = dataslicing(data=data, levels=n_levels)
    print(len(eegdata_sliced)) #21
    grouped = []
    for block in range(len(eegdata_sliced)):
        if transform == 'DHT':
            functiondata = dataDHT(eegdata_sliced[block])
        elif transform == 'DFT':
            functiondata = dataDFT(eegdata_sliced[block])
        datapool = datapooling(functiondata, axis=2, width=band_width)
        print(datapool.shape) #
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=2)
    print(groupeddata.shape)
    norm_groupeddata = NormalizeData(groupeddata)
    
    
    # mapping the labels 
    flattened = labels.flatten()
    encoding_values = LabelEncoder().fit_transform(flattened)
    mapped_labels = encoding_values.reshape(labels.shape)
    
    return norm_groupeddata, mapped_labels