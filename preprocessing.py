import jax.numpy as jnp
from slicing import dataslicing
from pooling import datapooling
from DHT import dataDHT
from normalization import NormalizeData
from sklearn.preprocessing import LabelEncoder

def dataprocessing(data, labels, n_levels=int, band_width=int):
    
    eegdata_sliced = dataslicing(data=data, levels=n_levels)

    grouped = []
    for block in range(len(eegdata_sliced)):
        dhtdata = dataDHT(eegdata_sliced[block])
        datapool = datapooling(dhtdata, axis=1, width=band_width)
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=1)
    norm_groupeddata = NormalizeData(groupeddata)
    
    # mapping the labels 
    flattened = labels.flatten()
    encoding_values = LabelEncoder().fit_transform(flattened)
    mapped_labels = encoding_values.reshape(labels.shape)
    
    return norm_groupeddata, mapped_labels