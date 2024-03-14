import jax.numpy as jnp
from main_functions.slicing import dataslicing
from main_functions.pooling import datapooling
from main_functions.DHT import dataDHT
from main_functions.utils import NormalizeData
from sklearn.preprocessing import LabelEncoder

def dataprocessing(data, labels, n_levels: int, band_width: int):
    eegdata_sliced = dataslicing(data=data, levels=n_levels)
    
    grouped = []
    for block in range(len(eegdata_sliced)):
        dhtdata = dataDHT(eegdata_sliced[block])
        datapool = datapooling(dhtdata, axis=1, width=band_width)
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=1)
    print(groupeddata.shape)
    norm_groupeddata = NormalizeData(groupeddata)
    
    # mapping the labels 
    flattened = labels.flatten()
    encoding_values = LabelEncoder().fit_transform(flattened)
    mapped_labels = encoding_values.reshape(labels.shape)
    
    return norm_groupeddata, mapped_labels