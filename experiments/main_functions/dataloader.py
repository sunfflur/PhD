import jax
import os
import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.io import loadmat,savemat
#from sklearn.model_selection import train_test_split
#import py7zr
from main_functions.utils import to_categorical
from main_functions.DHT import dataDHT
np.int = int
np.bool = bool
np.object = object
from braindecode.datasets import BNCI2014001
from braindecode.preprocessing import create_windows_from_events

seed=11

def filtro_CAR(X, axis=0):
  return X - np.mean(X,axis=axis,keepdims=True)


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
  #df = channel_df[channel_df['label'].isin(classes)]
  channel_df.loc[channel_df['label'].isin(range(0, 10)), 'label'] = classes[1]
  channel_df.loc[channel_df['label'].isin([-1]), 'label'] = classes[0]
  df = channel_df
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
  #testdf = channel_testdf[channel_testdf['label'].isin(classes)]
  channel_testdf.loc[channel_testdf['label'].isin(range(0, 10)), 'label'] = classes[1]
  channel_testdf.loc[channel_testdf['label'].isin([-1]), 'label'] = classes[0]
  testdf = channel_testdf
  # Convert to NumPy arrays
  test_data = jnp.array(testdf.drop(columns=['label']))
  test_labels = jnp.array(testdf['label'])

  
  x_train = filtro_CAR(x_data.reshape(x_data.shape[0], x_data.shape[1]//400, -1, 1), axis=1) #200,16,400,1
  #x_train = x_train - np.mean(x_train, axis=1, keepdims=True)
  y_train = to_categorical(labels, n_classes=len(classes))
  
  x_test = filtro_CAR(test_data.reshape(test_data.shape[0], test_data.shape[1]//400, -1, 1), axis=1)
  #x_test = x_test - np.mean(x_test, axis=1, keepdims=True)
  y_test = to_categorical(test_labels, n_classes=len(classes))
  
  return x_train, x_test, y_train, y_test
  
def BCIC_dataloader(subject, trial_start_offset_seconds, trial_stop_offset_seconds, path):
  os.environ['MNE_DATA'] = path
  subject_id = subject
  dataset = BNCI2014001(subject_ids=[subject_id])
  
  raw = dataset.datasets[0].raw

  # Get the channel names
  channel_names = raw.info['ch_names']
  #print(f'Channel names: {channel_names}')

  # Get only the signals, ignoring the targets
  signal_channels = [ch for ch in channel_names if not ch.startswith(('EOG1', 'EOG2', 'EOG3', 'stim'))]
  #print(f'Signal channels: {signal_channels}')

  # Create a new raw object with only signal channels
  #raw_signals = raw.copy().pick(signal_channels)
  
  #trial_start_offset_seconds = trial_start_offset_seconds #-0.5
  # Extract sampling frequency, check that they are same in all datasets
  sfreq = raw.info["sfreq"]
  assert all([ds.raw.info["sfreq"] == sfreq for ds in dataset.datasets])
  
  # Calculate the window start offset in samples.
  trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)
  trial_stop_offset_samples = int(trial_stop_offset_seconds * sfreq)

  # Create windows using braindecode function for this. It needs parameters to
  # define how windows should be used.
  windows_dataset = create_windows_from_events(
      dataset,
      trial_start_offset_samples=trial_start_offset_samples,
      trial_stop_offset_samples=trial_stop_offset_samples,
      preload=True,
      picks=signal_channels
  )
     
  splitted = windows_dataset.split("session")
  train_set = splitted["session_T"]  # Session train
  test_set = splitted["session_E"]  # Session evaluation
  
  train_data = []
  test_data = []
  for i in range(len(train_set)):
    x, y, z = train_set[i]
    m, n, p = test_set[i]
    train_data.append(x)
    test_data.append(m)
  
  x_train = filtro_CAR(np.expand_dims(np.array(train_data), axis=3), axis=1)
  y_train = to_categorical(train_set.get_metadata().target.values, n_classes=4)
  x_test = filtro_CAR(np.expand_dims(np.array(test_data), axis=3), axis=1)
  y_test = to_categorical(test_set.get_metadata().target.values, n_classes=4)
  
  return x_train, x_test, y_train, y_test

def MotionSense_dataloader(path):

  # Get path for standardized data
  train_path = path + 'train.csv'
  val_path = path + 'validation.csv'
  test_path = path + 'test.csv'

  # Read standardized data
  train_data = pd.read_csv(train_path)
  validation_data = pd.read_csv(val_path)
  test_data = pd.read_csv(test_path)

  # Filter accel and gyro attributes for training
  att = ('accel', 'gyro')
  filtered_train = [col for col in train_data.columns if col.startswith(att)]
  train_attributes = train_data[filtered_train]
  train_attributes_array = np.array(train_attributes).reshape(train_attributes.shape[0], 6, 60, 1)
  train_labels = to_categorical(train_data['standard activity code'].values, n_classes=6)


  # Filter accel and gyro attributes for validation
  filtered_val = [col for col in validation_data.columns if col.startswith(att)]
  val_attributes = validation_data[filtered_val]
  val_attributes_array = np.array(val_attributes).reshape(val_attributes.shape[0], 6, 60, 1)
  val_labels = to_categorical(validation_data['standard activity code'].values, n_classes=6)

  # Filter accel and gyro attributes for testing
  filtered_test = [col for col in test_data.columns if col.startswith(att)]
  test_attributes = test_data[filtered_test]
  test_attributes_array = np.array(test_attributes).reshape(test_attributes.shape[0], 6, 60, 1)
  test_labels = to_categorical(test_data['standard activity code'].values, n_classes=6)

  return train_attributes_array, train_labels, val_attributes_array, val_labels, test_attributes_array, test_labels