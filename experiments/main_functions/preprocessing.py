import jax.numpy as jnp
from main_functions.slicing import dataslicing
from main_functions.pooling import datapooling
from main_functions.DHT import dataDHT, dataDHTflip, dataDHT_half, dataDHT_halfsym, dataDHT_halfsym_1
from main_functions.DFT import dataDFT, dataDFT_half
from main_functions.utils import NormalizeData_, NormalizeData
from main_functions.windowing import windowing
from sklearn.preprocessing import LabelEncoder


def create_labels(data):
    a,freq,b,trial = data.shape
    return [[x for x in range(freq)] for _ in range(trial)]

def get_correct_data(data, label, num_trial=6):
    x_data = []
    y_data = []
    trial_number = []
    for trial in range(len(label)): #label tem tamanho trial*janelamento (ex 6*4=24)
        for l in label[trial]: #label[trial] acessa label que tem tamanho trial*janelamento na posicao da variavel trial que vai ser um vetor [0,1,2,3,4,5]
            x_data.append(data[:,l,:,trial]) #pega o dado com todos eletrodos e amostras de tempo no idx da frequencia l e trial
            y_data.append(l) #appenda na mesma posicao do x_data, o label de frequencia q ele pegou ali em cima
            trial_number.append(trial%num_trial) #appenda no gabarito o trial
    tx = list(map(lambda x: x.reshape(1,x.shape[0],x.shape[1]), x_data)) #cada elemento tx eh do shape (13,500) e tem trial*janelas*freqs elementos na lista
    
    return jnp.concatenate(tx, axis=0), jnp.array(y_data), trial_number

def dataprocessing(data, sampling_frequency: int, n_levels: int, band_width: int, transform: str, window: int, overlap:int, pooling_type: str):
    dataw = windowing(data, sampling_frequency=sampling_frequency, window=window, overlap=overlap)
    eegdata_sliced = dataslicing(data=dataw, levels=n_levels)
    #print(len(eegdata_sliced)) #21
    grouped = []
    for block in range(len(eegdata_sliced)):
        if transform == 'DHT':
            functiondata = dataDHT(eegdata_sliced[block])
        elif transform == 'DFT':
            functiondata = dataDFT(eegdata_sliced[block])
        datapool = datapooling(functiondata, axis=2, width=band_width, pooling_type=pooling_type)
        #print(datapool.shape) #
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=2)
    print("grouped_data:", groupeddata.shape)
    norm_groupeddata = NormalizeData(groupeddata)  # groupeddata (16, 4, 1498, 12)
    
    # mapping labels 
    creating_labels = create_labels(dataw)
    tx, mapped_labels, trial_number = get_correct_data(norm_groupeddata, creating_labels)
    
    return tx, mapped_labels, trial_number #(144, 13, 500), (144,)

def mnist_preprocessing(data, sampling_frequency: int, n_levels: int, band_width: int, transform: str, window: int, overlap: int, pooling_type: str):
    dataw = windowing(data, sampling_frequency=sampling_frequency, window=window, overlap=overlap)
    eegdata_sliced = dataslicing(data=dataw, levels=n_levels)
    #print(len(eegdata_sliced)) #21
    grouped = []
    for block in range(len(eegdata_sliced)):
        if transform == 'DHT':
            functiondata = dataDHT(eegdata_sliced[block])
        elif transform == 'DFT':
            functiondata = dataDFT(eegdata_sliced[block])
        datapool = datapooling(functiondata, axis=2, width=band_width, pooling_type=pooling_type)
        #print(datapool.shape) #
        grouped.append(datapool)
    groupeddata = jnp.concatenate(grouped, axis=2)
    norm_groupeddata = NormalizeData(groupeddata)  # groupeddata (16, 4, 1498, 12)
    norm_groupeddatar = norm_groupeddata.reshape(norm_groupeddata.shape[0], -1)
    return norm_groupeddatar
    