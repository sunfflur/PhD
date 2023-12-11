import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat,savemat
import py7zr
seed=11



def dataloader(subject, electrode, stimulus_frequency, trial, path):
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

  start = 250 #250
  end = -250 #-250
  
  sinais = []
  labels = []
  for f in stimulus_frequency:
    #print('freq:', f)
    #encontra o índice da frequência selecionada
    ii = np.where(freqs==f)

    #isola os registros de EEG dos L eletrodos daquele sujeito, para aquela freq. de estímulo e para aquela sessão
    if type(trial) == int:
      #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],trial],(64,1500-(start-end),1)))
      EEG = np.reshape(data[:,start:end,ii[1],trial],(64,1500-(start-end),1))
    else:
      #EEG = filtro_CAR(np.reshape(data[:,start:end,ii[1],:],(64,1500-(start-end),6)))
      EEG = np.reshape(data[:,start:end,ii[1],:],(64,1500-(start-end),6))

    #isola os registros de EEG dos L eletrodos daquele sujeito para aquela sessão
    #EEG_ss = np.reshape(data[:,:,:,trial],(64,1500))
    #dic = {"data": EEG_ss}
    #savemat('EEG.mat',dic)

    if type(electrode) == int:
      #índice correspondente ao eletrodo que queremos analisar (61 = eletrodo Oz, 55 = POz)
      id_ele = electrode #61
      #captura do sinal de EEG
      sinal = EEG[id_ele,:]
      label = jnp.ones((1, sinal.shape[1]))*f
      sinais.append(sinal)
      labels.append(label)
    else:
      #sinal = EEG[:,:]
      sinal = EEG[list(electrode),:]
      label = jnp.ones((sinal.shape[0], sinal.shape[2]))*f
      sinais.append(sinal)
      labels.append(label) 
  return jnp.concatenate(sinais), jnp.concatenate(labels)