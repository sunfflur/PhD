import jax.numpy as jnp
from experiments.main_functions.slicing import dataslicing
from experiments.main_functions.pooling import datapooling
from experiments.main_functions.DHT import dataDHT
from experiments.main_functions.DFT import dataDFT
from experiments.main_functions.utils import NormalizeData
from sklearn.preprocessing import LabelEncoder

def dataprocessing(data, labels, n_levels: int, band_width: int, transform: str):
    eegdata_sliced = dataslicing(data=data, levels=n_levels)
    
    grouped = []
    for block in range(len(eegdata_sliced)):
        if transform == 'DHT':
            functiondata = dataDHT(eegdata_sliced[block])
        elif transform == 'DFT':
            functiondata = dataDFT(eegdata_sliced[block])
        datapool = datapooling(functiondata, axis=1, width=band_width)
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=1)
    print(groupeddata.shape)
    norm_groupeddata = NormalizeData(groupeddata)
    
    # mapping the labels 
    flattened = labels.flatten()
    encoding_values = LabelEncoder().fit_transform(flattened)
    mapped_labels = encoding_values.reshape(labels.shape)
    
    return norm_groupeddata, mapped_labels