import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.io import loadmat,savemat
from sklearn.model_selection import train_test_split
import py7zr
from main_functions.utils import NormalizeData, to_categorical
from main_functions.DHT import dataDHT

seed=11

def filtro_CAR(X):
  return X - np.mean(X,axis=0,keepdims=True)


def dataloader(subject, electrode, stimulus_frequency, trial, sec_off, path):
  #escolha do sujeito cujos dados vamos processar/analisar
  
  suj = subject
  #zip_file_path = path+'S'+ str(suj) + ".mat.7z"
  filename = path + "S" + str(suj) + ".mat"
  
  """
  
  # Extract the .mat file from the zip archive
  with py7zr.SevenZipFile(zip_file_path, 'r') as z:
    z.extract(filename)
  """
  #escolha da sessão
  #trial = trial

  #escolha da frequência de estímulo (pelo valor em Hz)
  #freq_est = stimulus_frequency

  #carrega o vetor de frequências de estímulo na ordem correspondente aos índices
  data_freq = loadmat(path+'Freq_Phase.mat')
  #isola o vetor de frequências
  freqs = data_freq['freqs']
  
  #carrega os dados de EEG
  data_eeg = loadmat(filename)
  data = data_eeg['data']

  start = int(sec_off * 250) #250
  end = int(sec_off * (-250)) #-250
  
  sinais = []
  labels = []
  for f in stimulus_frequency:
    #encontra o índice da frequência selecionada
    ii = np.where(np.isclose(freqs,f))
    #isola os registros de EEG dos L eletrodos daquele sujeito, para aquela freq. de estímulo e para aquela sessão
    if type(trial) == int:
      #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],trial],(64,1500-(start-end),1)))
      if start != 0:
        EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],trial],(64, 1, 1500-(start-end),1)))
      else:
        EEG = filtro_CAR(np.reshape(data[:,:,ii[1],trial],(64, 1, 1500-(start-end),1)))
    else:
      #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],:],(64,1500-(start-end),6)))
      if start != 0:
        EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],:],(64, 1, 1500-(start-end),6)))
      else:
        EEG = filtro_CAR(np.reshape(data[:,:,ii[1],:],(64, 1, 1500-(start-end),6)))
      
      """      car = []
      for t in range(EEG_.shape[3]):
        EEG__ = EEG_[:,:,:,t].reshape(64, 1, 1500-(start-end), 1)
        #print(EEG__.shape)
        fcar = filtro_CAR(EEG__)
        fcar = fcar.reshape(fcar.shape[0], fcar.shape[1], fcar.shape[2], 1)
        car.append(fcar)
      EEG = jnp.concatenate(car, axis=3)"""

    #isola os registros de EEG dos L eletrodos daquele sujeito para aquela sessão
    #EEG_ss = np.reshape(data[:,:,:,trial],(64,1500))
    #dic = {"data": EEG_ss}
    #savemat('EEG.mat',dic)
    
    if type(electrode) == int:
      #índice correspondente ao eletrodo que queremos analisar (61 = eletrodo Oz, 55 = POz)
      id_ele = electrode #61
      #captura do sinal de EEG
      sinal = EEG[id_ele,:].reshape(1,1,EEG.shape[2],1)#[0]
      label = jnp.ones((sinal.shape[0],1))*f
      sinais.append(sinal)
      labels.append(label)
    elif type(electrode) == str:
      sinal = EEG#[:,:] #64,1,1500,1
      label = jnp.ones((sinal.shape[1], sinal.shape[3]))*f
      sinais.append(sinal)
      labels.append(label) 
    else:
      #sinal = EEG[:,:]
      sinal = EEG[list(electrode),:] #N, 1, 1500, 1

      label = jnp.ones((sinal.shape[1], sinal.shape[3]))*f
      sinais.append(sinal)
      labels.append(label) 

  sinal_concat = jnp.concatenate(sinais, axis=1)
  labels_concat = jnp.concatenate(labels, axis=0)
  return sinal_concat , labels_concat


def mnist_dataloader(path, electrodes, classes: list):
  train_mnist = pd.read_csv(path+"train.csv")
  test_mnist = pd.read_csv(path+"test.csv")
  
  #sets = [train_mnist, test_mnist]
  
  # ignoring the train images
  filtered_columns = [col for col in train_mnist.columns if not col.startswith('label_image')]
  filtered_df = train_mnist[filtered_columns]
  
  # selecting electrodes/channels
  filtered_channel = [col for col in filtered_df.columns if col.startswith(electrodes)]
  channel_df = filtered_df[filtered_channel]
  
  #getting data from labels 0 and 1 or 0 to 9
  df = channel_df[channel_df['label'].isin(classes)]
  
  # Convert to NumPy arrays
  x_data = jnp.array(df.drop(columns=['label']))
  labels = jnp.array(df['label'])
    
  # ignoring the test images
  filtered_testcolumns = [col for col in test_mnist.columns if not col.startswith('label_image')]
  filtered_testdf = test_mnist[filtered_testcolumns]
  # selecting electrodes/channels
  filtered_testchannel = [col for col in filtered_testdf.columns if col.startswith(electrodes)]
  channel_testdf = filtered_testdf[filtered_testchannel]
  #getting data from labels 0 and 1
  testdf = channel_testdf[channel_testdf['label'].isin(classes)]
  # Convert to NumPy arrays
  test_data = jnp.array(testdf.drop(columns=['label']))
  test_labels = jnp.array(testdf['label'])

  
  x_train = x_data.reshape(x_data.shape[0], x_data.shape[1]//400, -1, 1) #200,16,400,1
  x_train = x_train - np.mean(x_train, axis=1, keepdims=True)
  y_train = to_categorical(labels, n_classes=len(classes))
  
  x_test = test_data.reshape(test_data.shape[0], test_data.shape[1]//400, -1, 1)
  x_test = x_test - np.mean(x_test, axis=1, keepdims=True)
  y_test = to_categorical(test_labels, n_classes=len(classes))
  
  
  # applying the DHT
  
  #x_dhtdata = NormalizeData(jax.jit(dataDHT, device=jax.devices("cpu")[0])(x_train)).reshape(x_train.shape[0], -1)
  #test_dhtdata = NormalizeData(jax.jit(dataDHT, device=jax.devices("cpu")[0])(x_test)).reshape(x_test.shape[0], -1)
  return x_train, x_test, y_train, y_test
    